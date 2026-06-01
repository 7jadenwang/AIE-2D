import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import time as T
import imageio.v2 as io
import os
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim_loss


imagesD=[]
imagesO=[]
imagesT=[]

folder_name = 'test_repro'
os.makedirs(folder_name, exist_ok=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Working Device:',device)


def intensityOptLoss(firstDoC, intermediateDoC, finalDoC, target): #as per MSEC
    firstLoss = torch.linalg.matrix_norm(firstDoC - 0.0* target,'fro')
    intermediateLoss = torch.linalg.matrix_norm(intermediateDoC - 0.77 * target,'fro')
    finalLoss = torch.linalg.matrix_norm(finalDoC - 0.91 * target,'fro')
    #FinalLoss=F.mse_loss(finalDoC, target)
    #return FinalLoss
    return firstLoss + intermediateLoss + finalLoss #+ FinalLoss


def _to_nchw(image):
    if image.dim() == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.dim() == 3:
        return image.unsqueeze(1)
    if image.dim() == 4:
        return image
    raise ValueError(f'Expected 2D, 3D, or 4D image tensor, got shape {tuple(image.shape)}')


def _ssim_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d).view(1, 1, window_size, window_size)


def foregroundSSIMCuringLoss(finalDoC, target, foreground_threshold=15/255, window_size=11, sigma=1.5, eps=1e-8):
    finalDoC = _to_nchw(finalDoC.clamp(0, 1))
    target = _to_nchw(target.clamp(0, 1)).to(device=finalDoC.device, dtype=finalDoC.dtype)
    _, _, height, width = finalDoC.shape
    window_size = min(window_size, height, width)
    window_size = window_size if window_size % 2 == 1 else window_size - 1
    window_size = max(window_size, 1)
    window = _ssim_window(window_size, sigma, finalDoC.device, finalDoC.dtype)
    pad = window_size // 2

    def blur(x):
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect') if pad > 0 else x
        return F.conv2d(x, window)

    # Union foreground follows the notebook check, but detaches prediction thresholding.
    foreground = ((target > foreground_threshold) | (finalDoC.detach() > foreground_threshold)).to(finalDoC.dtype)
    mu_x, mu_y = blur(finalDoC), blur(target)
    sigma_x = blur(finalDoC * finalDoC) - mu_x ** 2
    sigma_y = blur(target * target) - mu_y ** 2
    sigma_xy = blur(finalDoC * target) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    foreground_pixels = foreground.sum()
    foreground_score = torch.where(
        foreground_pixels > 0,
        (ssim_map * foreground).sum() / foreground_pixels.clamp_min(eps),
        ssim_map.mean(),
    )
    return 1 - foreground_score


#Experimental Physical data
dx,dy=float(7.395e-6),float(7.395e-6)
dfsvty=float(2000e-12) #m2^2/s 2000um2/s
#dfsvty=float(200e-12) #O2 concentration-dependent
TEMPO_dfsvty=float(400e-12) #m2^2/s, TEMPO diffusion coefficient 400um2
# TEMPO_dfsvty unknown, it should be small. 
# PROBLEM: CANNOT be too small to create Gaussian kernel? 
# What if it is smaller than 1 pixel?

intensity=20 #mW/cm2 
#Change intensity with different data pls
dt=float(0.1) #s, time step
#0.2 for 5fps
total_steps=int(18/dt)
tstepT0 = int(1.0 / dt) # only for loss and optimization.
tstepT1 = int(14.0 / dt) # When epoch is 1 for the simulation, Loss does not matter
tstepT2 = int(16.5 / dt) # But need to change with DoC profile with distinct intensity

#O2inhibition=O2_inhibition_time * intensity #mJ/cm2 
O2inhibition=20.0538
# 0 for no O2 inhibition
#10.452 for 0mmol TEMPO concentration O2 only
#Total_inhibition_time=4.239 # from experimental data
Totalinhibtion=89.8478
#0 for no inhibition
#39.9671 for 1mmol TEMPO concentration
#89.8478 for 5mmol TEMPO concentration


#TEMPO_inibition_Time=Total_inhibition_time - O2_inhibition_time
#TEMPOinhibition=TEMPO_inibition_Time * intensity #mJ/cm2
TEMPOinhibition=max(0.0,Totalinhibtion - O2inhibition) 
#mJ/cm2 #clip = clamp

img=Image.open('./optl_Swiss.png')
print(f'Image mode:{img.mode}')
# now the target is 16-bit. 
# Dont convert to mode L to decrease the bit level
if img.mode == 'I;16': #16-bit
    target=np.asarray(img)
    max_val=2**16-1
elif img.mode == 'L': #8-bit
    target=np.asarray(img)
    max_val=255
else: #RGB/RGBA
    target=np.asarray(img.convert('L'))
    max_val=255
target=(target/max_val).astype(np.float32)
#target=(target/255).astype(np.float32) # convert to float32 for processing
#plt.imshow(target,cmap='gray')
#plt.show()


H,W=target.shape
mask=torch.tensor(target.copy() * 255, dtype=torch.float32, device=device) # scale to [0,255] to match /255 in physics
opt_mask=torch.nn.Parameter(mask.clone()) #shape(H,W)

#Swiss O2diff convo
O2_sigma=(2*dfsvty*dt)**0.5
O2_sigma=O2_sigma/dx
print(f'''O2 diffusion sigma: {O2_sigma:.2f} pixels''')
if O2_sigma<1:
    print("Warning: O2 diffusion is too small.")
    #quit()
O2_kernel_size=int((O2_sigma-0.8)/0.3+1)*2+1
print(f'O2 kernel size: {O2_kernel_size}')
O2_kernel=cv2.getGaussianKernel(O2_kernel_size,O2_sigma)
O2_diff=torch.from_numpy(np.outer(O2_kernel,O2_kernel)).view(1,1,O2_kernel_size,O2_kernel_size).to(torch.float32).to(device)
O2_pad=O2_kernel_size//2

#Swiss TEMPOdiff convo
TEMPO_sigma=(2*TEMPO_dfsvty*dt)**0.5 #wrong here
TEMPO_sigma=TEMPO_sigma/dx
print(f'''TEMPO diffusion sigma: {TEMPO_sigma:.2f} pixels''')
if TEMPO_sigma<1:
    print("Warning: TEMPO diffusion is too small.")
    #quit()
TEMPO_kernel_size=int((TEMPO_sigma-0.8)/0.3+1)*2+1
print(f'TEMPO kernel size: {TEMPO_kernel_size}')
TEMPO_kernel=cv2.getGaussianKernel(TEMPO_kernel_size,TEMPO_sigma)
TEMPO_diff=torch.from_numpy(np.outer(TEMPO_kernel,TEMPO_kernel)).view(1,1,TEMPO_kernel_size,TEMPO_kernel_size).to(torch.float32).to(device)
TEMPO_pad=TEMPO_kernel_size//2

#Light Scattering Gaussian Blur convo
blur_size=600e-6
ls_kernel_size=int(blur_size/dx) if int(blur_size/dx)%2!=0 else int(blur_size/dx)+1
ls_sigma=0.3*((ls_kernel_size-1)*0.5-1)+0.8
print(f'scattering  sigma: {ls_sigma:.2f} pixels')
print(f'scattering kernel size: {ls_kernel_size}')
ls_kernel=cv2.getGaussianKernel(ls_kernel_size,ls_sigma)
ls=torch.from_numpy(np.outer(ls_kernel,ls_kernel)).view(1,1,ls_kernel_size,ls_kernel_size).to(torch.float32).to(device)
ls_pad=ls_kernel_size//2

numEpochs=1
#if epoch is 1, it just simulate without optimization
optimizer=torch.optim.Adam([opt_mask],lr=0.77)
loss_history=[]
MidpointDoC=[]
MidpointO2=[]
MidpointTEMPO=[]

for epoch in range(numEpochs):
    #scattering every time before curing start
    #opt_mask_pre=opt_mask.view(1,1,H,W)
    opt_mask_pre=opt_mask.unsqueeze(0).unsqueeze(0)
    opt_mask_padded=F.pad(opt_mask_pre,pad=(ls_pad,ls_pad,ls_pad,ls_pad),mode='reflect')
    blur_mask=F.conv2d(opt_mask_padded,ls)[0,0]
    #blur_mask=opt_mask # for optimization without scattering
    

    #plt.imshow(blur_mask.detach().cpu().numpy(),cmap='gray')
    #plt.show()
    O2=[(torch.ones((H,W))*(O2inhibition)).to(torch.float32).to(device)]
    TEMPO=[(torch.ones((H,W))*(TEMPOinhibition)).to(torch.float32).to(device)]
    Dose=[torch.zeros((H,W)).to(torch.float32).to(device)]
    DoC=[torch.zeros((H,W)).to(torch.float32).to(device)]

    #A = -0.0231*(blur_mask.clamp(min=1e-12)/255 * intensity) + 2.044
    B = 0.0290*(blur_mask.clamp(min=1e-12)/255 * intensity) + 0.2101
    #C=O2inhibition/(blur_mask.clamp(min=1e-12)/255*intensity)
    #print(B[H//2,W//2].item()) # for debug

    # absorption coefficient, mJ/cm2
    tic=T.time()

    for step in range(total_steps):
        # For O2 diffusion
        O2_pre=O2[-1].view(1,1,H,W)
        O2_padded=F.pad(O2_pre,pad=(O2_pad,O2_pad,O2_pad,O2_pad),mode='reflect')
        O2_diffused=F.conv2d(O2_padded,O2_diff)[0,0]
        O2_diffused=O2[-1] #For local O2 with no diffusion

        # For TEMPO diffusion
        TEMPO_pre=TEMPO[-1].view(1,1,H,W)
        TEMPO_padded=F.pad(TEMPO_pre,pad=(TEMPO_pad,TEMPO_pad,TEMPO_pad,TEMPO_pad),mode='reflect')
        TEMPO_diffused=F.conv2d(TEMPO_padded,TEMPO_diff)[0,0]
        #TEMPO_diffused=TEMPO[-1] #For local TEMPO with no diffusion

        energy=(blur_mask.clamp(min=1e-12)/255)*intensity*dt
        
        O2next=torch.clamp(O2_diffused-energy, min=0)
        O2.append(O2next)
        TEMPOnext=torch.where(O2next<=0, torch.clamp(TEMPO_diffused-energy, min=0), TEMPO_diffused)
        TEMPO.append(TEMPOnext)
        #print(step) if O2next.min()<=0 else None
        
        # Tim_accmulation_Method
        Dosenext = torch.where((O2next<=0) & (TEMPOnext<=0), Dose[-1]+energy-O2_diffused-TEMPO_diffused, Dose[-1])
        Dose.append(Dosenext)
        t=Dosenext/(blur_mask.clamp(min=1e-12)/255*intensity)
    
        #DoCnext= 1-torch.exp(-B*(t-C).clamp(min=0))
        DoCnext=torch.where((O2next<=0) & (TEMPOnext<=0), 1-torch.exp(-(B*t).clamp(min=0)), DoC[-1])
        #DoCnext=torch.where((O2next<=0) & (TEMPOnext<=0), 1-torch.exp(B*(-t)), DoC[-1])
        
        '''
        # Jaden_Step_accumulation_Method (at most 1 step error)
        #DoCnext=torch.where((O2next > 0), DoC[-1], 1-(1-DoC[-1])*torch.exp(-A*dt)) # just for O2-only
        DoCnext=torch.where((O2next<=0) & (TEMPOnext<=0), 1-(1-DoC[-1])*torch.exp(-A*dt), DoC[-1])
        DoCnext.clamp_(min=0,max=1) # O2>0 means no cure can start
        #DoCnext = DoCnext - O2next * 0.005 + 0.005
        '''
        DoC.append(DoCnext)
        if epoch==numEpochs-1:
            if step%1==0:
                DoCprint = DoCnext #DoC range [0,1]
                DoCprint.data.clamp_(min = 0)
                DoCprint = DoCprint.detach().cpu().numpy() * 255 # for 8-bit image
                DoCprint = DoCprint.astype(dtype = np.uint8)
                io.imwrite(os.path.join(folder_name,f'DoC_{str(step)}.png'),DoCprint)
                imagesD.append(io.imread(os.path.join(folder_name,f'DoC_{str(step)}.png')))

                figO, axO = plt.subplots()
                imageO = axO.matshow(O2_diffused.detach().cpu().numpy())
                cbarO = figO.colorbar(imageO)
                imageO.set_clim(0,O2inhibition)
                plt.figtext(0,0,f't={str(step*dt)}s at {str(step)}th O2')
                file_path = os.path.join(folder_name, f'O2_{str(step)}.png')
                plt.savefig(file_path,bbox_inches='tight')
                imagesO.append(io.imread(file_path))
                plt.close(figO)

                figT, axT = plt.subplots()
                imageT = axT.matshow(TEMPO_diffused.detach().cpu().numpy())
                cbarT = figT.colorbar(imageT)
                imageT.set_clim(0,TEMPOinhibition)
                plt.figtext(0,0,f't={str(step*dt)}s at {str(step)}th TEMPO')
                file_path = os.path.join(folder_name, f'TEMPO_{str(step)}.png')
                plt.savefig(file_path,bbox_inches='tight')
                imagesT.append(io.imread(file_path))
                plt.close(figT)

                #Plot midpoint DoC (Only for circle)
                MidpointDoC.append(DoCnext[H//2,W//2].item())
                MidpointO2.append(O2next[H//2,W//2].item()) #
                MidpointTEMPO.append(TEMPOnext[H//2,W//2].item()) #



   
    #Loss=foregroundWeightedBCELoss(DoC[tstepT2], target=(mask/255)).to(device)
    #Loss=cornerWeightedShapeLoss(DoC[tstepT2], target=(mask/255)).to(device)
    #Loss=foregroundSSIMCuringLoss(DoC[tstepT2], target=(mask/255)).to(device)
    Loss=intensityOptLoss(DoC[tstepT0], DoC[tstepT1], DoC[tstepT2], target=(mask/255)).to(device)
    #SML=ssim_loss((DoC[-1]>(15/255)).view(1,1,H,W), ((mask/255)>(15/255)).view(1,1,H,W),data_range=1.0).to(device)
    #Loss=1-SML
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {Loss.item():.4f}, Time per epoch: {T.time()-tic:.4f} seconds')
        
        
    loss_history.append(Loss.item())
    if numEpochs == 1: continue  # simulation-only: skip optimization
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    opt_mask.data.clamp_(0,255)

plt.figure()
plt.plot(np.arange(len(loss_history)),loss_history)
plt.savefig(os.path.join(folder_name,'aaa_loss_history.png'))
#plt.show()
final_opt_mask=(opt_mask.detach().cpu().numpy())/255*65535
#final_opt_mask is 16bit
#plt.imshow(final_opt_mask,cmap='gray')
#plt.show()
file_path = os.path.join(folder_name, 'aaa_final_opt_mask.png')
io.imwrite(file_path, final_opt_mask.astype(np.uint16))
blur_mask_pre=opt_mask.unsqueeze(0).unsqueeze(0)
blur_mask_padded=F.pad(blur_mask_pre,pad=(ls_pad,ls_pad,ls_pad,ls_pad),mode='reflect')
final_blur_mask=F.conv2d(blur_mask_padded,ls)[0,0].detach().cpu().numpy()
final_blur_mask=final_blur_mask/255*65535
#final_blur_mask is 16bit
#plt.imshow(final_blur_mask,cmap='gray')
#plt.show()
file_path = os.path.join(folder_name, 'aaa_final_blur_mask.png')
io.imwrite(file_path, final_blur_mask.astype(np.uint16))

#Evaluation of final result
file_path = os.path.join(folder_name, 'allDoC.gif')
io.mimsave(file_path, imagesD, format='GIF', loop=0, fps = 100)
file_path = os.path.join(folder_name, 'allO2.gif')
io.mimsave(file_path, imagesO, format='GIF', loop=0, fps = 500)
file_path = os.path.join(folder_name, 'allTEMPO.gif')
io.mimsave(file_path, imagesT, format='GIF', loop=0, fps = 500)

#Report Midpoint info

MidpointDoC_arr = np.array(MidpointDoC)
t0=next((i*dt for i,c in enumerate(MidpointDoC_arr) if c >= 0.001), None)
t30 = next((i*dt for i,c in enumerate(MidpointDoC_arr) if c >= 0.30), None)
t90 = next((i*dt for i,c in enumerate(MidpointDoC_arr) if c >= 0.90), None)
print(f'Midpoint DoC starts at t={t0}s')
print(f'Midpoint DoC reaches 30% at t={t30}s, 90% at t={t90}s')


MidpointO2_arr = np.array(MidpointO2)
#print(MidpointO2_arr[0:30]-MidpointO2_arr[1:31])
tO2_step=next((i for i,c in enumerate(MidpointO2_arr) if c <=0),None)
tO2=dt*tO2_step
print(f'O2 depleted at t={tO2} s')


MidpointTEMPO_arr = np.array(MidpointTEMPO)
#print(MidpointTEMPO_arr[20:50]-MidpointTEMPO_arr[21:51])
tTEMPO_step=next((i for i,c in enumerate(MidpointTEMPO_arr) if c <=0),None)
tTEMPO=dt*tTEMPO_step
print(f'TEMPO depleted at t={tTEMPO} s')

plt.figure()
Inhibition_steps=int((max(tO2,tTEMPO))/dt)
file_path=os.path.join(folder_name,'aaa_MidpointIE_Curve.png')
plt.plot(np.arange(Inhibition_steps+5)*dt,
         MidpointO2_arr[:Inhibition_steps+5]
         +MidpointTEMPO_arr[:Inhibition_steps+5])
plt.xlim(0,(Inhibition_steps+(5/dt))*dt)
plt.xticks(np.arange(0,(Inhibition_steps+(5/dt))*dt,1))
#plt.show()
plt.savefig(file_path)

plt.figure()
file_path=os.path.join(folder_name,'aaa_MidpointDoC.png')
plt.plot(np.arange(len(MidpointDoC_arr))*dt, MidpointDoC_arr*100)
#np.asarray() share memory. while np.array() create a copy (safer)
plt.xlabel('Time (s)')
plt.ylabel('Midpoint DoC (%)')
plt.xlim(0,15)
plt.xticks(np.arange(0, 15, 1))
plt.savefig(file_path)
