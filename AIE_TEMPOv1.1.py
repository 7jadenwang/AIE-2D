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

#folder_name = 'test_repro'
folder_name ='260825\\120mW_5mMol\\260825_120mW_5mMol_Sync_rect_5s_Opt'
save_path=os.path.join('.\\',folder_name)
#save_path=os.path.join('.\\260722_circles_TPEoac\\LShape_Simulations',folder_name)
os.makedirs(save_path, exist_ok=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Working Device:',device)


def intensityOptLoss(firstDoC, intermediateDoC, finalDoC, target): #as per MSEC
    firstLoss = torch.linalg.matrix_norm((firstDoC - 0.0* target),'fro')
    intermediateLoss = torch.linalg.matrix_norm((intermediateDoC - 0.77 * target),'fro')
    finalLoss = torch.linalg.matrix_norm((finalDoC - 0.40 * target),'fro')
    #FinalLoss=F.mse_loss(finalDoC, target)
    return finalLoss#+intermediateLoss
    #return firstLoss + intermediateLoss + finalLoss


def _harris_corner_weights(
    target, corner_window=5, corner_radius=3, harris_k=0.04, eps=1e-8
):
    if corner_window < 1 or corner_window % 2 == 0:
        raise ValueError('corner_window must be an odd positive integer')
    if corner_radius < 0:
        raise ValueError('corner_radius must be nonnegative')

    image = target.detach().unsqueeze(0).unsqueeze(0)
    sobel_x = image.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ).view(1, 1, 3, 3) / 8
    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_x.transpose(-1, -2), padding=1)
    pad = corner_window // 2
    tensor_terms = [
        F.avg_pool2d(term, corner_window, stride=1, padding=pad)
        for term in (gx * gx, gy * gy, gx * gy)
    ]
    sxx, syy, sxy = tensor_terms
    response = (sxx * syy - sxy.square()) - harris_k * (sxx + syy).square()
    response = response.clamp_min(0)
    response = response / response.amax().clamp_min(eps)
    if corner_radius:
        size = 2 * corner_radius + 1
        response = F.max_pool2d(response, size, stride=1, padding=corner_radius)
    return response[0, 0]


def intensityOptLoss_v2(
    firstDoC, intermediateDoC, finalDoC, target, corner_weight=1.0,
    corner_window=11, corner_radius=4, harris_k=0.04, eps=1e-8
):
    target = target.to(device=finalDoC.device, dtype=finalDoC.dtype)
    base_loss = intensityOptLoss(firstDoC, intermediateDoC, finalDoC, target)
    weights = _harris_corner_weights(
        target, corner_window, corner_radius, harris_k, eps
    )
    residual = finalDoC - 0.91 * target
    weighted_mse = (weights * residual.square()).sum()
    corner_loss = torch.sqrt(weighted_mse / weights.sum().clamp_min(eps) + eps)
    has_corners = (weights.sum() > eps).to(finalDoC.dtype)
    return base_loss + corner_weight * has_corners * corner_loss


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
dx,dy=float(7.6e-6),float(7.6e-6)
#dx,dy=float(4.905e-6),float(4.905e-6) # for 432x468 DLP

blur_size=30e-6 # used to be 600um to 7.4um pxs
 # set it zeros to optimize without scattering
O2_dfsvty=float(400e-12) #m2^2/s 2000um2/s
#dfsvty=float(200e-12) #O2 concentration-dependent
TEMPO_dfsvty=float(400e-12) #m2^2/s, TEMPO diffusion coefficient 400um2
#The TEMPO now is still too small for diffusion.
# PROBLEM: CANNOT be too small to create Gaussian kernel? 
# What if it is smaller than 1 pixel?

intensity=120 #mW/cm2
#Change intensity with different data pls
chainGrowth_noise_std=0.0 #relative std of quenched per-pixel randomness in local cure rate B

dt=float(0.05) #s, time step
#0.2 for 5fps
total_steps=int(5.5/dt)
tstepT0 = int(0.2 / dt) # only for loss and optimization.
tstepT1 = int(2.0 / dt) # When epoch is 1 for the simulation, Loss does not matter
tstepT2 = int(5.0 / dt)  # But need to change with DoC profile with distinct intensity

#O2inhibition=O2_inhibition_time * intensity #mJ/cm2 
O2inhibition=27.7117
# 0 for no O2 inhibition
#10.452 for 0mmol TEMPO concentration O2 only
#Total_inhibition_time=4.239 # from experimental data
Totalinhibtion=119.7295
#0 for no TEMPO inhibition
#51.7456 for 1mmol TEMPO concentration
#119.7295 for 5mmol TEMPO concentration


#TEMPO_inibition_Time=Total_inhibition_time - O2_inhibition_time
#TEMPOinhibition=TEMPO_inibition_Time * intensity #mJ/cm2
TEMPOinhibition=max(0.0,Totalinhibtion - O2inhibition)
#mJ/cm2 #clip = clamp

img=Image.open('./GEO/sync_rect.png')
img.save(f'./{folder_name}/aaa_target.png')
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
target=(target/max_val).astype(np.float32) #Normalize to [0,1]

#plt.imshow(target,cmap='gray')
#plt.show()


H,W=target.shape
DoC_radius=25
mask=torch.tensor(target.copy() * 255, dtype=torch.float32, device=device) # scale to [0,255] to match /255 in physics
opt_mask=torch.nn.Parameter(mask.clone()) #shape(H,W)

grayscale_floor=15.0  #Zak needs it
# min opt_mask value enforced inside the cure zone, so cured pixels never rely 100% on scatter
#How will if affect? PENDING
cure_zone=mask>grayscale_floor # define a target fre ground.

#Swiss O2diff convo
O2_sigma=(2*O2_dfsvty*dt)**0.5
O2_sigma=O2_sigma/dx
print(f'''O2 diffusion sigma: {O2_sigma:.2f} pixels''')
if O2_sigma<1:
    print("Warning: O2 diffusion is too small.")
    #quit()
O2_kernel_size=int((O2_sigma-0.8)/0.3+1)*2+1
print(f'O2 kernel size: {O2_kernel_size}')
O2_kernel=cv2.getGaussianKernel(O2_kernel_size,O2_sigma) #set very small values to 0
O2_diff=torch.from_numpy(np.outer(O2_kernel,O2_kernel)).view(1,1,O2_kernel_size,O2_kernel_size).to(torch.float32).to(device)
print(O2_diff)
O2_pad=O2_kernel_size//2

#Swiss TEMPOdiff convo
TEMPO_sigma=(2*TEMPO_dfsvty*dt)**0.5
TEMPO_sigma=TEMPO_sigma/dx
print(f'''TEMPO diffusion sigma: {TEMPO_sigma:.2f} pixels''')
if TEMPO_sigma<1:
    print("Warning: TEMPO diffusion is too small.")
    #quit()
TEMPO_kernel_size=int((TEMPO_sigma-0.8)/0.3+1)*2+1
print(f'TEMPO kernel size: {TEMPO_kernel_size}')
TEMPO_kernel=cv2.getGaussianKernel(TEMPO_kernel_size,TEMPO_sigma*0.8) #smaller sigma -> slower diffusion -> extreme situation: local TEMPO
TEMPO_diff=torch.from_numpy(np.outer(TEMPO_kernel,TEMPO_kernel)).view(1,1,TEMPO_kernel_size,TEMPO_kernel_size).to(torch.float32).to(device)
print(TEMPO_diff)
TEMPO_pad=TEMPO_kernel_size//2

#Light Scattering Gaussian Blur convo
ls_kernel_size=int(blur_size/dx) if int(blur_size/dx)%2!=0 else int(blur_size/dx)+1
ls_sigma=0.3*((ls_kernel_size-1)*0.5-1)+0.8
print(f'scattering  sigma: {ls_sigma:.2f} pixels')
print(f'scattering kernel size: {ls_kernel_size}')
ls_kernel=cv2.getGaussianKernel(ls_kernel_size,ls_sigma)
#print(ls_kernel)
ls=torch.from_numpy(np.outer(ls_kernel,ls_kernel)).view(1,1,ls_kernel_size,ls_kernel_size).to(torch.float32).to(device)
ls_pad=ls_kernel_size//2

#Gradient smoothing kernel (optimization aid when blur_size==0; independent of physical scattering)
grad_smooth_sigma=2.0 # unit in pixels
grad_smooth_kernel_size=int(grad_smooth_sigma*6)|1
grad_smooth_kernel_np=cv2.getGaussianKernel(grad_smooth_kernel_size,grad_smooth_sigma)
grad_smooth_kernel=torch.from_numpy(np.outer(grad_smooth_kernel_np,grad_smooth_kernel_np)).view(1,1,grad_smooth_kernel_size,grad_smooth_kernel_size).to(torch.float32).to(device)
grad_smooth_pad=grad_smooth_kernel_size//2

numEpochs=1000
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
    


    if numEpochs == 1:
        #plt.imshow(blur_mask.detach().cpu().numpy(),cmap='gray')
        #plt.show()
        print(f'Intensity at the Center: {blur_mask[H//2-DoC_radius:H//2+DoC_radius,
                                                    W//2-DoC_radius:W//2+DoC_radius].mean().item():.4f}')
    O2=[(torch.ones((H,W))*(O2inhibition)).to(torch.float32).to(device)]
    TEMPO=[(torch.ones((H,W))*(TEMPOinhibition)).to(torch.float32).to(device)]
    Dose=[torch.zeros((H,W)).to(torch.float32).to(device)]
    DoC=[torch.zeros((H,W)).to(torch.float32).to(device)]

    #A = -0.0231*(blur_mask.clamp(min=1e-12)/255 * intensity) + 2.044
    #B = 0.0133*(blur_mask.clamp(min=1e-12)/255 * intensity) + 0.4638 #0mMTEMPO
    #B =0.0152*(blur_mask.clamp(min=1e-12)/255 * intensity) + 0.3135 #1mMTEMPO
    B =0.0069*(blur_mask.clamp(min=1e-12)/255 * intensity) + 0.3815 #5mMTEMPO
    if chainGrowth_noise_std > 0:
        B_noise=(1 + chainGrowth_noise_std * torch.randn(H, W, device=device)).clamp(min=1e-3)
        B = B * B_noise

    #C=O2inhibition/(blur_mask.clamp(min=1e-12)/255*intensity)
    #print(B[H//2,W//2].item()) # for debug

    # absorption coefficient, mJ/cm2
    tic=T.time()

    for step in range(total_steps):
        # For O2 diffusion
        O2_pre=O2[-1].view(1,1,H,W)
        O2_padded=F.pad(O2_pre,pad=(O2_pad,O2_pad,O2_pad,O2_pad),mode='reflect')
        O2_diffused=F.conv2d(O2_padded,O2_diff)[0,0]
        #O2_diffused=O2[-1] #For local O2 with no diffusion

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
        if epoch==numEpochs-1: # for the final epoch
            if step%2==0:
                DoCprint = DoCnext #DoC range [0,1]
                DoCprint.data.clamp_(min = 0)
                DoCprint = DoCprint.detach().cpu().numpy() * 255 # for 8-bit image
                DoCprint = DoCprint.astype(dtype = np.uint8)
                io.imwrite(os.path.join(save_path,f'DoC_{str(step)}.png'),DoCprint)
                imagesD.append(io.imread(os.path.join(save_path,f'DoC_{str(step)}.png')))
                
                if O2next[H//2-DoC_radius:H//2+DoC_radius,W//2-DoC_radius:W//2+DoC_radius].mean().item()>=0:
                    figO, axO = plt.subplots()
                    imageO = axO.matshow(O2_diffused.detach().cpu().numpy())
                    cbarO = figO.colorbar(imageO)
                    imageO.set_clim(0,O2inhibition)
                    plt.figtext(0,0,f't={str(step*dt)}s at {str(step)}th O2')
                    file_path = os.path.join(save_path, f'O2_{str(step)}.png')
                    plt.savefig(file_path,bbox_inches='tight')
                    imagesO.append(io.imread(file_path))
                    plt.close(figO)



                if TEMPOnext[H//2-DoC_radius:H//2+DoC_radius,W//2-DoC_radius:W//2+DoC_radius].mean().item()>=0:
                    figT, axT = plt.subplots()
                    imageT = axT.matshow(TEMPO_diffused.detach().cpu().numpy())
                    cbarT = figT.colorbar(imageT)
                    imageT.set_clim(0,TEMPOinhibition)
                    plt.figtext(0,0,f't={str(step*dt)}s at {str(step)}th TEMPO')
                    file_path = os.path.join(save_path, f'TEMPO_{str(step)}.png')
                    plt.savefig(file_path,bbox_inches='tight')
                    imagesT.append(io.imread(file_path))
                    plt.close(figT)

                #Plot midpoint DoC (Only for circle)
                MidpointDoC.append(DoCnext[H//2-DoC_radius:H//2+DoC_radius,W//2-DoC_radius:W//2+DoC_radius].mean().item())
                
                MidpointO2.append(O2next[H//2-DoC_radius:H//2+DoC_radius,W//2-DoC_radius:W//2+DoC_radius].mean().item())
                
                MidpointTEMPO.append(TEMPOnext[H//2-DoC_radius:H//2+DoC_radius,W//2-DoC_radius:W//2+DoC_radius].mean().item()) #


   
    #Loss=foregroundWeightedBCELoss(DoC[tstepT2], target=(mask/255)).to(device)
    #Loss=cornerWeightedShapeLoss(DoC[tstepT2], target=(mask/255)).to(device)
    #Loss=foregroundSSIMCuringLoss(DoC[tstepT2], target=(mask/255)).to(device)
    Loss=intensityOptLoss(DoC[tstepT0], DoC[tstepT1], DoC[tstepT2], target=(mask/255)).to(device)
    #Loss=intensityOptLoss_v2(DoC[tstepT0], DoC[tstepT1], DoC[tstepT2], target=(mask/255)).to(device)
    #SML=ssim_loss((DoC[-1]>(15/255)).view(1,1,H,W), ((mask/255)>(15/255)).view(1,1,H,W),data_range=1.0).to(device)
    #Loss=1-SML
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {Loss.item():.4f}, Time per epoch: {T.time()-tic:.4f} seconds')
        
        
    loss_history.append(Loss.item())
    if numEpochs == 1: continue  # simulation-only: skip optimization
    optimizer.zero_grad()
    Loss.backward()
    if blur_size==0:
        with torch.no_grad():
            grad_padded=F.pad(opt_mask.grad.view(1,1,H,W),pad=(grad_smooth_pad,grad_smooth_pad,grad_smooth_pad,grad_smooth_pad),mode='reflect')
            opt_mask.grad=F.conv2d(grad_padded,grad_smooth_kernel)[0,0]
    optimizer.step()
    opt_mask.data.clamp_(0,255)
    opt_mask.data[cure_zone]=opt_mask.data[cure_zone].clamp(min=grayscale_floor)


#opt_mask.data range from o to 255
plt.figure()
plt.plot(np.arange(len(loss_history)),loss_history)
plt.savefig(os.path.join(save_path,'aaa_loss_history.png'))
#plt.show()
final_opt_mask=(opt_mask.detach().cpu().numpy())
#final_opt_mask is 16bit
#plt.imshow(final_opt_mask,cmap='gray')
#plt.show()
file_path = os.path.join(save_path, 'aaa_final_opt_mask.png')
io.imwrite(file_path, final_opt_mask.astype(np.uint8))
blur_mask_pre=opt_mask.unsqueeze(0).unsqueeze(0)
blur_mask_padded=F.pad(blur_mask_pre,pad=(ls_pad,ls_pad,ls_pad,ls_pad),mode='reflect')
final_blur_mask=F.conv2d(blur_mask_padded,ls)[0,0].detach().cpu().numpy()
final_blur_mask=final_blur_mask
#plt.imshow(final_blur_mask,cmap='gray')
#plt.show()
file_path = os.path.join(save_path, 'aaa_final_blur_mask.png')
io.imwrite(file_path, final_blur_mask.astype(np.uint8))

#Evaluation of final result
file_path = os.path.join(save_path, 'allDoC.gif')
io.mimsave(file_path, imagesD, format='GIF', loop=0, fps = 100)
file_path = os.path.join(save_path, 'allO2.gif')
io.mimsave(file_path, imagesO, format='GIF', loop=0, fps = 500)
file_path = os.path.join(save_path, 'allTEMPO.gif')
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
file_path=os.path.join(save_path,'aaa_MidpointIE_Curve.png')
plt.plot(np.arange(Inhibition_steps+5)*dt,
         MidpointO2_arr[:Inhibition_steps+5]
         +MidpointTEMPO_arr[:Inhibition_steps+5])
plt.xlim(0,(Inhibition_steps+(5/dt))*dt)
plt.xticks(np.arange(0,(Inhibition_steps+(5/dt))*dt,1))
#plt.show()
plt.savefig(file_path)

plt.figure()
file_path=os.path.join(save_path,'aaa_MidpointDoC.png')
plt.plot(np.arange(len(MidpointDoC_arr))*dt, MidpointDoC_arr*100)
#np.asarray() share memory. while np.array() create a copy (safer)
plt.xlabel('Time (s)')
plt.ylabel('Midpoint DoC (%)')
plt.xlim(0,15)
plt.xticks(np.arange(0, 15, 1))
plt.savefig(file_path)
