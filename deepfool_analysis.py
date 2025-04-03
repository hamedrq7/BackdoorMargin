import argparse 
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.metrics import accuracy_score

# relative import hacks (sorry)
import os 
import sys 
import inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user

from log_utils import (
    Logger,
    Timer,
    Constants,
    DefaultList,
)
import loading_utils
from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet, LeNet
from log_utils import make_dir

parser = argparse.ArgumentParser(description="Analysing results")
parser.add_argument(
    "--save-dir", type=str, default="", metavar="N", help="save directory of results"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA use"
)

cmds = parser.parse_args()
print(type(cmds))
runPath = cmds.save_dir

sys.stdout = Logger("{}/analyse.log".format(runPath))
args = torch.load(runPath + "/args.rar", weights_only=False)


# cuda stuff
needs_conversion = cmds.no_cuda and args.cuda
conversion_kwargs = {"map_location": lambda st, loc: st} if needs_conversion else {}
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

loading_utils.set_reproducability(args.seed)

print("\n# load dataset: %s " % args.dataset)
dataset_train, args.nb_classes, mean, std = build_poisoned_training_set(is_train=True, args=args)
dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) 

if args.model == 'badnet': 
    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
elif args.model == 'lenet': 
    model = LeNet().to(device)
model.load_state_dict(torch.load(f'{runPath}/checkpoints/{args.model}.pth', weights_only=False), strict=True,)
print(model)


from attacks.DeepFool import DeepFool
from attacks.DeepFoolTargeted import DeepFoolTargeted

if __name__ == "__main__":
    with Timer("BackDoor-FUM: base_analyse.py") as t:
        # test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        # print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        # print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")

        test_ds = {'clean': data_loader_val_clean, 'poison': data_loader_val_poisoned}
       
        for phase in ['clean',]: #  'poison' 
            for steps in [50]:
                for overshoot in [0.02, 0.05, 0.1]: # 0.02,  
                    batch_idx = 0
                    df = DeepFool(model, steps=steps, overshoot=overshoot)
                    # dft = DeepFoolTargeted(model, steps=100, )
                    # target_class = args.trigger_label
                    # imgs1 = []
                    # imgs2 = []
                    # lbls1 = []
                    # lbls2 = []
                    all_labels = []
                    all_adv_preds = []
                    all_cln_preds = []
                    all_deltas = []
                    for img, label in tqdm(test_ds[phase]): 
                        out_cln = model(img.to(device))
                        cln_pred = torch.argmax(out_cln, dim=1)
                        adv_images, __adv_pred, delta = df(img, label, )
                        # adv_images, __adv_pred, _ = dft(img, label, attack_targets=torch.ones_like(label)*target_class)
                        out_adv = model(adv_images)
                        adv_pred = torch.argmax(out_adv, dim=1)
                        all_labels.append(label.cpu().numpy())
                        all_cln_preds.append(cln_pred.cpu().numpy())
                        all_adv_preds.append(adv_pred.cpu().numpy())
                        all_deltas.append(delta.cpu().numpy())
                        batch_idx += 1

                        if batch_idx >= 25: 
                            break
                    
                    all_deltas    = np.concatenate(all_deltas)
                    all_labels    = np.concatenate(all_labels)
                    all_cln_preds = np.concatenate(all_cln_preds)
                    all_adv_preds = np.concatenate(all_adv_preds)
                    print(steps, overshoot, 'acc', accuracy_score(all_labels, all_adv_preds), f'(cln: {accuracy_score(all_labels, all_cln_preds)}')
                    
                    l2_norms = np.linalg.norm(all_deltas.reshape(all_deltas.shape[0], -1), ord=2, axis=1)
                    linf_norms = np.linalg.norm(all_deltas.reshape(all_deltas.shape[0], -1), ord=np.inf, axis=1)

                    print('L2', np.mean(l2_norms), np.max(l2_norms), np.min(l2_norms))
                    print('LInf', np.mean(linf_norms), np.max(linf_norms), np.min(linf_norms))
                    # from plot_utils import plot_and_save_confusion_matrix
                    # plot_and_save_confusion_matrix(all_labels, all_cln_preds, f'{runPath}/DeepFool', name=f'DS_{phase} cln')
                    # plot_and_save_confusion_matrix(all_labels, all_adv_preds, f'{runPath}/DeepFool', name=f'DS_{phase} adv')

        # for phase in ['clean',]: #  'poison' 
        #     batch_idx = 0
        #     df = DeepFool(model, steps=100, )
        #     dft = DeepFoolTargeted(model, steps=100, )
        #     target_class = args.trigger_label
            
        #     imgs1 = []
        #     lbls1 = []
            
        #     imgs2 = []
        #     lbls2 = []
        #     deltas2 = []
            
        #     imgs3 = []
        #     lbls3 = []
        #     deltas3 = []
            
        #     for img, label in tqdm(test_ds[phase]): 
        #         out_cln = model(img.to(device))
        #         cln_pred = torch.argmax(out_cln, dim=1)
        #         imgs1.append(img[0:8].numpy())
        #         lbls1.append(cln_pred[0:8].cpu().numpy())
                
        #         adv_images, adv_pred, deltas = df(img, label)
        #         imgs2.append(adv_images[0:8].cpu().numpy())
        #         deltas2.append(deltas[0:8].cpu().numpy())
        #         lbls2.append(adv_pred[0:8].cpu().numpy())
                
        #         adv_images, adv_pred, deltas = dft(img, label, attack_targets=torch.ones_like(label)*target_class)
        #         imgs3.append(adv_images[0:8].cpu().numpy())
        #         deltas3.append(deltas[0:8].cpu().numpy())
        #         lbls3.append(adv_pred[0:8].cpu().numpy())
                
        #         break

        #     imgs1 = np.concatenate(imgs1)
        #     lbls1 = np.concatenate(lbls1)

        #     imgs2 = np.concatenate(imgs2)
        #     deltas2 = np.concatenate(deltas2)
        #     lbls2 = np.concatenate(lbls2)
            
        #     imgs3 = np.concatenate(imgs3)
        #     deltas3 = np.concatenate(deltas3)
        #     lbls3 = np.concatenate(lbls3)
            
        #     from plot_utils import display_image_grid
        #     display_image_grid(images1=imgs1, images2=imgs2, titles1=lbls1, titles2=lbls2, save_path=f'{runPath}/images', name_to_save='cln-df_img')    
        #     display_image_grid(images1=imgs1, images2=deltas2, titles1=lbls1, titles2=lbls2, save_path=f'{runPath}/images', name_to_save='cln-df_noise')    

        #     display_image_grid(images1=imgs1, images2=imgs3, titles1=lbls3, titles2=lbls3, save_path=f'{runPath}/images', name_to_save='cln-dft_img')    
        #     display_image_grid(images1=imgs1, images2=deltas3, titles1=lbls3, titles2=lbls3, save_path=f'{runPath}/images', name_to_save='cln-dft_noise')    

        #     display_image_grid(images1=deltas2, images2=deltas3, titles1=lbls2, titles2=lbls3, save_path=f'{runPath}/images', name_to_save='df_noise-dft_noise')    

