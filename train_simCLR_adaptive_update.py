import argparse
from torch import clamp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet_multi_bn_bottleneck import resnet18, proj_head
from utils import *
import torchvision.transforms as transforms
from torch.nn import DataParallel
from StrategyNet import ResNet18_Strategy
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from data.cifar10 import CustomImagenet100, CustomCIFAR10 , CustomCIFAR100
from optimizer.lars import LARS
import wandb
from torch.nn import DataParallel 
import torch.backends.cudnn as cudnn


def setup_deterministic_cuda():
    if torch.cuda.is_available():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
def to_list(a):
    return list(map(int,a.split(",")))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torchvision.set_random_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--experiment', type=str, help='location for saving trained models')
parser.add_argument('--data', type=str, default='../../data', help='location of the data')
parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to be used, (cifar10 or cifar100)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', action='store_true', help='if resume training')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--step_factor', default=1, type=int, help="Multiplicative factor for total_steps")
parser.add_argument('--ACL_DS', action='store_true', help='if specified, use pgd dual mode,(cal both adversarial and clean)')
parser.add_argument('--twoLayerProj', action='store_true', help='if specified, use two layers linear head for simclr proj head')
parser.add_argument('--pgd_iter', default=5, type=int, help='how many iterations employed to attack the model')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight for adversarial accuracy in reward computation')
parser.add_argument('--interval_num', type=int, default=10, help='Interval for updating the strategy network')
parser.add_argument('--epsilon_types', type=list, default=list(range(3,16)), help='Possible epsilon values for attack') #3,16
parser.add_argument('--attack_iters_types', type=list, default=list(range(3,15)), help='Possible number of attack iterations') #3,15
parser.add_argument('--step_size_types', type=list, default=list(range(1,6)), help='Possible step sizes for attack') #1,6
parser.add_argument('--policy_model_lr', type=float, default=0.001, help='Learning rate for policy model')
parser.add_argument('--r1',type=float,default=1,help="Weight of representation similarity")
parser.add_argument('--r2',type=float,default=1,help="Weight of adv loss in reward")
parser.add_argument('--r3',type=float,default=1,help="Weight of clean loss in reward")


parser.add_argument('--sim_weight', type=float, default=1, help='Weight for similarity loss in main model training')
parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm for strategy network')
parser.add_argument('--gpu_ids',type=to_list,default="3,4", help="Gpu ids for DataParallel")

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=1).mean()


def worker_init_fn(worker_id):
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    #########################
    # Setting deterministic cuda
    setup_deterministic_cuda()
    #########################
    global args
    args = parser.parse_args()
    global device
    device=torch.device(f"cuda:{args.gpu_ids[0]}")
    
    ###################
    # Setting Seed
    set_seed(args.seed)
    ###################
    
    # Initialize a new run
    wandb.init(project=args.experiment)
    assert args.dataset in ['cifar100', 'cifar10', 'imagenet100']

    save_dir = os.path.join('checkpoints', args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    log = logger(path=save_dir)
    log.info(str(args))
    setup_seed(args.seed)

    # different attack corresponding to different bn settings
    if not args.ACL_DS:
        bn_names = ['normal', ]
    else:
        bn_names = ['normal', 'pgd']

    # define model
    model = resnet18(pretrained=False, bn_names=bn_names)
    
    ### Strategy model
    strategy_model = ResNet18_Strategy(args).to(torch.device(f"cuda:{args.gpu_ids[0]}"))
    strategy_model = nn.DataParallel(strategy_model.to(device),device_ids=args.gpu_ids)

    ch = model.fc.in_features
    model.fc = proj_head(ch, bn_names=bn_names, twoLayerProj=args.twoLayerProj)
    model.to(torch.device(f"cuda:{args.gpu_ids[0]}"))
    model=nn.DataParallel(model,device_ids=args.gpu_ids)
    wandb.watch(model)
    cudnn.benchmark = True

    strength = 1.0
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength)
    rnd_gray = transforms.RandomGrayscale(p=0.2 * strength)
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(1.0 - 0.9 * strength, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])
    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    global mu,std, upper_limit,lower_limit
    # dataset process
    if args.dataset == 'cifar10':
        train_datasets = CustomCIFAR10(root=args.data, train=True, transform=tfs_train, download=True)
        val_train_datasets = datasets.CIFAR10(root=args.data, train=True, transform=tfs_test, download=True)
        test_datasets = datasets.CIFAR10(root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 10
        cifar10_mean = (0.0, 0.0, 0.0)
        cifar10_std = (1.0, 1.0, 1.0)
        mu = torch.tensor(cifar10_mean).view(3,1,1).to(device)
        std = torch.tensor(cifar10_std).view(3,1,1).to(device)

        upper_limit = ((1 - mu)/ std)
        lower_limit = ((0 - mu)/ std)

    elif args.dataset == 'cifar100':
        train_datasets = CustomCIFAR100(root=args.data, train=True, transform=tfs_train, download=True)
        val_train_datasets = datasets.CIFAR100(root=args.data, train=True, transform=tfs_test, download=True)
        test_datasets = datasets.CIFAR100(root=args.data, train=False, transform=tfs_test, download=True)
        num_classes = 100
        cifar100_mean = (0.0, 0.0, 0.0)
        cifar100_std = (1.0, 1.0, 1.0)
        mu = torch.tensor(cifar100_mean).view(3,1,1).to(device)
        std = torch.tensor(cifar100_std).view(3,1,1).to(device)

        upper_limit = ((1 - mu)/ std)
        lower_limit = ((0 - mu)/ std)

    elif args.dataset == 'imagenet100':
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8 )
        rnd_gray = transforms.RandomGrayscale(p=0.2 )
        tfs_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.08, 1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomApply([Solarization()], p=0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )
        tfs_test = transforms.Compose([
            transforms.Resize(256),  # resize shorter
            transforms.CenterCrop(224),  # take center crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
        ])
        train_datasets = CustomImagenet100(transform=tfs_train)
        val_train_datasets = datasets.ImageFolder("IM_100_Data_Train_Data_Path",transform=tfs_test)
        test_datasets = datasets.ImageFolder("IM_100_Data_Val_Data_Path",transform=tfs_test)
        num_classes = 100
        imagenet100_mean = (0.485, 0.456, 0.406)
        imagenet100_std = (0.228, 0.224, 0.225)
        mu = torch.tensor(imagenet100_mean).view(3,1,1).to(device)
        std = torch.tensor(imagenet100_std).view(3,1,1).to(device)

        upper_limit = ((1 - mu)/ std)
        lower_limit = ((0 - mu)/ std)
    else:
        cifar10_mean = (0.0, 0.0, 0.0)
        cifar10_std = (1.0, 1.0, 1.0)
        mu = torch.tensor(cifar10_mean).view(3,1,1).to(device)
        std = torch.tensor(cifar10_std).view(3,1,1).to(device)

        upper_limit = ((1 - mu)/ std)
        lower_limit = ((0 - mu)/ std)
        print("unknow dataset")
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed))

    val_train_loader = torch.utils.data.DataLoader(
        val_train_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed))

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        num_workers=4,
        batch_size=args.batch_size,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        strategy_model_optimizer = torch.optim.Adam(strategy_model.parameters(), lr=args.policy_model_lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
        strategy_model_optimizer = LARS(strategy_model.parameters(), lr=args.policy_model_lr, weight_decay=1e-6)
        
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6, momentum=0.9)
        strategy_model_optimizer = torch.optim.SGD(strategy_model.parameters(), lr=args.policy_model_lr, weight_decay=1e-6, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10 * args.step_factor, ], gamma=1)
        strategy_scheduler = torch.optim.lr_scheduler.MultiStepLR(strategy_model_optimizer, milestones=[args.epochs * len(train_loader) * 10 * args.step_factor, ], gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs*len(train_loader)*args.step_factor,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=10 * len(train_loader))
        )
        strategy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            strategy_model_optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs*len(train_loader)*args.step_factor // args.interval_num,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.policy_model_lr,
                                                    warmup_steps=10 * len(train_loader) // args.interval_num)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    global current_step
    current_step=0

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model_both.pt'))
            if 'state_dict' in checkpoint:
                new_state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    new_key = 'module.' + k if not k.startswith('module.') else k
                    new_state_dict[new_key] = v
                checkpoint['state_dict'] = new_state_dict
                model.load_state_dict(checkpoint['state_dict'])
                #strategynet
                new_state_dict = {}
                for k, v in checkpoint['strategy_model'].items():
                    new_key = 'module.' + k if not k.startswith('module.') else k
                    new_state_dict[new_key] = v
                checkpoint['strategy_model'] = new_state_dict
                strategy_model.load_state_dict(checkpoint['strategy_model'])
                
            else:
                model.load_state_dict(checkpoint)

        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            #strategynet
            strategy_model_optimizer.load_state_dict(checkpoint['strategy_optim'])
             
            for i in range((start_epoch - 1) * len(train_loader)):
                current_step+=1
                scheduler.step()
            
            for i in range((start_epoch - 1) * len(train_loader)//args.interval_num):
                strategy_scheduler.step()
            log.info("resume the checkpoints for main and strategy models {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False
    
    for epoch in range(start_epoch, args.epochs + 1):

        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        wandb.log({"lr":optimizer.state_dict()['param_groups'][0]['lr']})
        train(train_loader, model,strategy_model, optimizer, strategy_model_optimizer, scheduler,strategy_scheduler, epoch, log)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optim': optimizer.state_dict(),
            'strategy_model': strategy_model.module.state_dict(),
            'strategy_optim': strategy_model_optimizer.state_dict(),            
        }, filename=os.path.join(save_dir, 'model_both.pt'))

        if epoch % 1000 == 0 and epoch > 0:
            # evaluate acc
            # (acc,r_acc), (tacc,r_tacc) = validate(val_train_loader, test_loader, model, log, num_classes=num_classes)
            # log.info('train_accuracy {acc:.3f}'
            #          .format(acc=acc))
            # log.info('test_accuracy {tacc:.3f}'
            #          .format(tacc=tacc))
            # log.info('robust_train_accuracy {acc:.3f}'
            #          .format(acc=r_acc))
            # log.info('robust_test_accuracy {tacc:.3f}'
            #          .format(tacc=r_tacc))

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optim': optimizer.state_dict(),
                'strategy_model': strategy_model.module.state_dict(),
                'strategy_optim': strategy_model_optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))



def pgd_attack(target_model, images, policies, args,current_step, sameBN=False):
    target_model.eval()
    X = images.to(device)
    batch_size = X.size(0)
    # what is std and lower limit? Probable
    torch.manual_seed(args.seed + current_step)
    delta_batch = torch.zeros_like(X).to(device)
    init_epsilon = (8 / 255.) / std
    for i in range(len(init_epsilon)):
        delta_batch[:, i, :, :].uniform_(-init_epsilon[i][0][0].item(), init_epsilon[i][0][0].item())
    delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
    delta_batch.requires_grad = True

    epsilon_batch = []
    alpha_batch = []
    attack_iters_batch = []

    for ii in range(batch_size):
        epsilon = (policies[ii][ii]['epsilon'] / 255.) / std
        epsilon_batch.append(epsilon.cpu().numpy())
        
        alpha = (policies[ii][ii]['step_size'] / 255.) / std
        alpha_batch.append(alpha.cpu().numpy())
        
        attack_iters = policies[ii][ii]['attack_iters']
        temp_batch = torch.randint(attack_iters, attack_iters + 1, (3, 1, 1))
        attack_iters_batch.append(temp_batch.cpu().numpy())

    alpha_batch = torch.from_numpy(np.array(alpha_batch)).to(device)
    epsilon_batch = torch.from_numpy(np.array(epsilon_batch)).to(device)
    attack_iters_batch = torch.from_numpy(np.array(attack_iters_batch)).to(device)

    max_attack_iters = torch.max(attack_iters_batch).cpu().numpy()

    for _ in range(max_attack_iters):
        mask_batch = attack_iters_batch.ge(1).float()
        

        if not sameBN:
            perturbed_features = target_model(X + delta_batch, 'normal')
        else:
            perturbed_features = target_model(X + delta_batch, 'pgd')
            # clean_features = target_model(X,'normal')
            
        # loss =  nt_xent(perturbed_features) - cosine_similarity(perturbed_features, clean_features)
        target_model.zero_grad()
        loss =  nt_xent(perturbed_features)
        loss.backward()
        grad = delta_batch.grad.detach()
        delta_batch.data = clamp(delta_batch + mask_batch * alpha_batch * torch.sign(grad), -epsilon_batch, epsilon_batch)
        delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)  
        attack_iters_batch = attack_iters_batch - 1
        delta_batch.grad.zero_()
        
        
    target_model.train()
    delta_batch = delta_batch.detach()
    return X + delta_batch


def compute_reward(target_model, clean_inputs, adv_inputs,strategy_model,acl_ds):
    target_model.eval()
    with torch.no_grad():
        if acl_ds:
            clean_features = target_model(clean_inputs,'normal') 
            adv_features = target_model(adv_inputs,'pgd')
        else:
            clean_features = target_model(clean_inputs,'normal') 
            adv_features = target_model(adv_inputs,'normal')
        # Original Image View not similar to Adversarial Image View

    # Compute contrastive losses
    clean_loss = nt_xent(clean_features)
    adv_loss = nt_xent(adv_features) # Pertubed Image 1 View 1 to be different from Pertubed Image 1 View 2 
    
    #reward = (args.r1*rep_sim + args.r2*adv_loss - args.r3*clean_loss) / 3
    if args.r1 > 0.0:
        sim_loss = nt_xent(torch.stack((clean_features, adv_features), dim=1).reshape(-1, 512))
        reward = (-args.r1*sim_loss + args.r2*adv_loss - args.r3*clean_loss) / 3
        wandb.log({"Similarity Loss":sim_loss,"Adverserial Loss":adv_loss,"Clean Loss":clean_loss},commit=False)
    else:
        reward = (args.r2*adv_loss - args.r3*clean_loss) / 3
        wandb.log({"Adverserial Loss":adv_loss,"Clean Loss":clean_loss},commit=False)
    #wandb.log({"Adverserial Loss":adv_loss,"Clean Loss":clean_loss},commit=False)
    strategy_model.module.saved_rewards.append(reward)
    return reward
    

    
def update_strategy_network(strategy_model, strategy_optimizer, strategy_scheduler):
    assert len(strategy_model.module.saved_log_probs) == len(strategy_model.module.saved_rewards), "Length of saved log probs and rewards should be the same"
    policy_loss = [log_prob*reward for log_prob,reward in zip(strategy_model.module.saved_log_probs,strategy_model.module.saved_rewards)]
    policy_loss = -torch.stack(policy_loss).sum()

    strategy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(strategy_model.parameters(), args.clip_grad_norm)
    wandb.log({"Policy_loss":policy_loss.detach().cpu().item()},commit=False)
    strategy_optimizer.step()
    strategy_scheduler.step()
    strategy_model.module.saved_log_probs = []
    strategy_model.module.saved_rewards = []
    


def select_action(strategy_model, state,args,current_step):
    outputs = strategy_model(state)
    actions = []
    log_probs = []
    for output in outputs:
        probs = F.softmax(output, dim=-1)
        m = Categorical(probs)
        torch.manual_seed(args.seed + current_step)
        action = m.sample()
        actions.append(action)
        log_probs.append(m.log_prob(action))
    
    strategy_model.module.saved_log_probs.append(sum(log_probs))
    wandb.log({"LogProbs": sum(log_probs)},commit=False)
    return actions



def convert_actions_to_policies(actions, args):
    policies = []
    # Probable error strategy net has 4 outputs --> Attack_method, Attack_epsilon, Attack_iters, Attack_step_size
    for j in range(len(actions[0])):
        policy = {
            'epsilon': args.epsilon_types[actions[0][j]],
            'attack_iters': args.attack_iters_types[actions[1][j]],
            'step_size': args.step_size_types[actions[2][j]]
        }
        Actions_Counts['epsilon'][args.epsilon_types[actions[0][j]]]+=1
        Actions_Counts['attack_iters'][args.attack_iters_types[actions[1][j]]]+=1
        Actions_Counts['step_size'][args.step_size_types[actions[2][j]]]+=1
        
        Action_Counts_step['epsilon_step'][args.epsilon_types[actions[0][j]]]+=1
        Action_Counts_step['attack_iters_step'][args.attack_iters_types[actions[1][j]]]+=1
        Action_Counts_step['step_size_step'][args.step_size_types[actions[2][j]]]+=1
        policies.append({j: policy})
    return policies

def train(train_loader, model, strategy_model, optimizer, strategy_optimizer, scheduler, strategy_scheduler, epoch, log):
    losses = AverageMeter()
    rewards = AverageMeter()
    clean_loss=AverageMeter()
    adv_loss=AverageMeter()
    sim_loss=AverageMeter()

    global Actions_Counts,Action_Counts_step
    Actions_Counts = {
        'epsilon': {eps:0 for eps in args.epsilon_types},
        'attack_iters': {iters:0 for iters in args.attack_iters_types},
        'step_size': {step_size:0 for step_size in args.step_size_types}
    }
    
    for i, (inputs) in enumerate(train_loader):
        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).to(device)
        global current_step
        current_step+=1
        #--------Training strategy model----------#
        if current_step % args.interval_num == 0 and current_step>0:
            strategy_model.train()
            model.eval()
            strategy_model.module.saved_log_probs = []
            strategy_model.module.saved_rewards = []
            temp_inputs=inputs.clone()
            actions = select_action(strategy_model, temp_inputs,args,current_step)
            policies = convert_actions_to_policies(actions, args)

            # Generate adversarial examples
            adv_temp_inputs = pgd_attack(model, temp_inputs, policies,args=args,current_step=current_step,sameBN=args.ACL_DS)
            reward = compute_reward(model, temp_inputs, adv_temp_inputs, strategy_model,args.ACL_DS)
            rewards.update(reward.item(), temp_inputs.size(0))
            update_strategy_network(strategy_model, strategy_optimizer, strategy_scheduler)


        #--------Training target model---------#
        strategy_model.eval()
        model.train()
        # Get actions from the strategy network
        Action_Counts_step = {
        'epsilon_step': {eps:0 for eps in args.epsilon_types},
        'attack_iters_step': {iters:0 for iters in args.attack_iters_types},
        'step_size_step': {step_size:0 for step_size in args.step_size_types}
        }
        strategy_model.module.saved_log_probs = []
        strategy_model.module.saved_rewards = []
        actions = select_action(strategy_model, inputs,args,current_step)
        policies = convert_actions_to_policies(actions, args)

        # Generate adversarial examples
        adv_inputs = pgd_attack(model, inputs, policies,args=args,current_step=current_step,sameBN=args.ACL_DS)

        # Train the main model
        # probable --> not using the dual BN part
        model.train()
        if args.ACL_DS:
            features_clean = model(inputs,'normal')
            features_adv = model(adv_inputs,'pgd')
        else:
            features_clean = model(inputs,'normal')
            features_adv = model(adv_inputs,'normal')
        # print(f"{features_adv.shape=},{features_clean.shape=}")

        # import pdb;pdb.set_trace()
        # Compute losses
        contrastive_loss_clean = nt_xent(features_clean)
        contrastive_loss_adv = nt_xent(features_adv)
        #similarity_loss = 1 - cosine_similarity(features_clean, features_adv)
        similarity_loss = nt_xent(torch.stack((features_clean, features_adv), dim=1).reshape(-1, 512))

        # Combined loss
        loss = (contrastive_loss_clean + contrastive_loss_adv) / 2 + args.sim_weight * similarity_loss
        wandb.log({"contrastive_loss_clear":contrastive_loss_clean,"contrainstive_loss_adv":contrastive_loss_adv,"similarity_loss":similarity_loss},commit=False)
        #wandb.log({"contrastive_loss_clear":contrastive_loss_clean,"contrainstive_loss_adv":contrastive_loss_adv},commit=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), inputs.size(0))
        clean_loss.update(contrastive_loss_clean.item(),inputs.size(0))
        adv_loss.update(contrastive_loss_adv.item(),inputs.size(0))
        sim_loss.update(similarity_loss.item(),inputs.size(0))
        
            

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Reward {reward.val:.3f} ({reward.avg:.3f})'.format(
                         epoch, i, len(train_loader), loss=losses, reward=rewards))
        
        wandb.log({"epoch": epoch, "loss_batch": losses.val, "rewards_batch": rewards.val}, commit=False)
         
    wandb.log(Actions_Counts,commit=False)
    wandb.log({"loss_epoch":losses.avg,"reward_epoch":rewards.avg,"contrastive_loss_epoch":clean_loss.avg,"contrastive_loss_adv":adv_loss.avg,"similarity_loss_epoch":sim_loss.avg})
    #wandb.log({"loss_epoch":losses.avg,"reward_epoch":rewards.avg,"contrastive_loss_epoch":clean_loss.avg,"contrastive_loss_adv":adv_loss.avg})
    
    return losses.avg, rewards.avg

def attack_pgd(target_model, inputs, targets, epsilon, alpha, attack_iters,sameBN=False):
    target_model.eval()
    X = inputs.to(device)
    Y = targets.to(device)
    batch_size = X.size(0)
    # what is std and lower limit? Probable
    torch.manual_seed(args.seed + current_step)
    delta_batch = torch.zeros_like(X).to(device)
    init_epsilon = (8 / 255.) / std
    for i in range(len(init_epsilon)):
        delta_batch[:, i, :, :].uniform_(-init_epsilon[i][0][0].item(), init_epsilon[i][0][0].item())
    delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)
    delta_batch.requires_grad = True
    for _ in range(attack_iters):
        if sameBN:
            perturbed_features = target_model(X + delta_batch, 'normal')
        else:
            perturbed_features = target_model(X + delta_batch, 'pgd')
            # clean_features = target_model(X,'normal')
            
        # loss =  nt_xent(perturbed_features) - cosine_similarity(perturbed_features, clean_features)
        target_model.zero_grad()
        loss =  F.cross_entropy(perturbed_features,Y)
        loss.backward()
        grad = delta_batch.grad.detach()
        delta_batch.data = clamp(delta_batch + alpha * torch.sign(grad), -epsilon, epsilon)
        delta_batch.data = clamp(delta_batch, lower_limit - X, upper_limit - X)  
        delta_batch.grad.zero_()
    target_model.train()
    delta_batch = delta_batch.detach()
    return X + delta_batch

def validate(train_loader, val_loader, model, log, num_classes=10):
    """
    Run evaluation
    """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train_time_meter = AverageMeter()
    losses = AverageMeter()
    losses.reset()
    end = time.time()

    # train a fc on the representation
    for param in model.parameters():
        param.requires_grad = False

    previous_fc = model.module.fc
    ch = model.module.fc.in_features
    model.module.fc = nn.Linear(ch, num_classes)
    model.to(device)

    epochs_max = 100
    lr = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                epochs_max*len(train_loader),
                                                1,  # since lr_lambda computes multiplicative factor
                                                1e-6 / lr,
                                                warmup_steps=0)
    )

    for epoch in range(epochs_max):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, (sample) in enumerate(train_loader):

            x, y = sample[0].to(device), sample[1].to(device)
            p = model.eval()(x, 'normal')
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(float(loss.detach().cpu()))

            train_time = time.time() - end
            end = time.time()
            train_time_meter.update(train_time)

        log.info('Test epoch: ({0})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'train_time: {train_time.avg:.2f}\t'.format(
                    epoch, loss=losses, train_time=train_time_meter))
        
    acc = []
    for loader in [train_loader, val_loader]:
        losses = AverageMeter()
        losses.reset()
        top1 = AverageMeter()
        losses_adv = AverageMeter()
        top1_adv = AverageMeter()

        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            adv_inputs = attack_pgd(model,inputs,targets,epsilon=(8 / 255.) / std,alpha=(2 / 255.) / std,attack_iters=10)
            # compute output
            with torch.no_grad():
                outputs = model.eval()(inputs, 'normal')
                outputs_adv = model.eval()(adv_inputs, 'pgd')
                loss = criterion(outputs, targets)
                loss_adv = criterion(outputs_adv,targets)

            outputs = outputs.float()
            outputs_adv = outputs_adv.float()
            loss = loss.float()
            loss_adv = loss_adv.float()

            # measure accuracy and record loss
            prec1 = accuracy(outputs.data, targets)[0]
            prec1_adv = accuracy(outputs_adv.data,targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            losses_adv.update(loss_adv.item(),inputs.size(0))
            top1_adv.update(prec1_adv.item(),inputs.size(0))

            if i % args.print_freq == 0:
                log.info('Test: [{0}/{1}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Loss_Adv {loss_adv.val:.4f} ({loss.avg:.4f})\t'
                         'Robust Accuracy {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                             i, len(loader), loss=losses, top1=top1, loss_adv=losses_adv, top1_adv=top1_adv))
        acc.append([top1.avg,top1_adv.avg])

    # recover every thing
    model.module.fc = previous_fc
    model.module.to(device)
    for param in model.parameters():
        param.requires_grad = True
    wandb.log({"Train_accuracy":acc[0][0],"Train_accuracy_adv":acc[0][1]})
    wandb.log({"Test_accuracy":acc[1][0],"Test_accuracy_adv":acc[1][1]})
    return acc


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)



        

if __name__ == '__main__':
    main()


