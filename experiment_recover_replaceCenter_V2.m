close all
clear
clc
addpath(genpath(pwd))
% load('Yeast_H15_Continue_FOV_step_1.2_Size_3000_NumImg1521_07272020.mat')
%%
close all
centerX=1550;centerY=1850;
nump=400;
imSeq_t2_recover = imSeq_t2(centerX-nump+1:centerX+nump,centerY-nump+1:centerY+nump,1:end);% 1500 200
% imSeq_t2_recover = imSeq_t2(1270:1269+512,1575:1574+512,:);
figure;imshow(imSeq_t2_recover(:,:,1),[]);
%%
% figure;
% for hj=1:imNum
%     imshow(imSeq_t2_recover(:,:,hj),[]);pause(0.05);
% end
%% setup the parameters for the lensless imaging system
%imSeq_t2_recover = imSeq_t2;
imSizeX = size(imSeq_t2_recover,1); %image size
imSizeY = size(imSeq_t2_recover,2); %image size
imNum = size(imSeq_t2_recover,3); %image number
waveLength = 0.532e-6; 
mag = 1; 
pixelSize0 = 1.85e-6; 
pixelSize = pixelSize0/mag; 
%%
load('loc_dftpc_XY.mat');
% ylocation=-pos1_X_stageFeedback/pixelSize0*1e-6;
% xlocation=pos1_Y_stageFeedback/pixelSize0*1e-6;
figure;plot(xlocation(1:100),ylocation(1:100),'r*-');hold on
%% shift raw images
imInitial_obj=zeros(imSizeX,imSizeY,'single');  
for i=1:imNum
    disp(i);
    imInitial_obj=imInitial_obj+abs(subpixelshift3GPU(sqrt(single(imSeq_t2_recover(:,:,i))),xlocation(i,1),ylocation(i,1))); %shift back
end
imInitial_obj=imInitial_obj/imNum;
imInitial_obj=gpuArray(imInitial_obj);
imInitial_objpad=imresize(imInitial_obj,[imSizeX*mag imSizeY*mag]);
imInitial_ptpad=imresize(sqrt(single(mean(imSeq_t2_recover,3))),[imSizeX*mag imSizeY*mag]);
figure;
subplot(121);imshow(abs(imInitial_objpad),[]);title('initial object pattern'); % show initial guess
subplot(122);imshow(abs(imInitial_ptpad),[]);title('initial padding pattern');
% %%
% load('E:\Zichao\Ecoli_0815_water\result\Ecoli_H5_1145_d2z835_3loops_pad1.mat','ObjectRecovery','PatternRecovery');
% imInitial_objpad=imresize(abs(ObjectRecovery(:,:)),[imSizeX*mag imSizeY*mag]);
% imInitial_ptpad=imresize(abs(PatternRecovery(:,:)),[imSizeX*mag imSizeY*mag]);
% % clear PatternRecover_y
% 
% figure;
% subplot(121);imshow(abs(imInitial_objpad),[]);title('initial object pattern'); % show initial guess
% subplot(122);imshow(abs(imInitial_ptpad),[]);title('initial padding pattern');
%% recovery process
clc
close all
imRandNum = randperm(imNum);
% imRandNum = 1:imNum;
z_d2 = (835).*1e-6; % 1150 the distance between the diffuser and the CCD
z_d1 = (396).*1e-6;% the distance between the diffuser and the sample 2090e-6-z_d2; 
LoopN1 = 3;
gamaO = 1;
gamaP = 1;
alphaP = 1;
alphaO = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kernal = ones(mag);
index0 = 1:mag:imSizeX*mag;
Xindex = index0;
for i = 2:mag
    Xindex(end+1:end+imSizeX) = index0 +(i-1);%[1:3:size(I_lowGPU,1),2:3:6,3:3:6]
end
inv_index0 = 1:imSizeX:imSizeX*mag;
inv_Xindex = inv_index0;
for i = 2:imSizeX
    inv_Xindex(end+1:end+mag) = inv_index0 +(i-1);%[1:3:size(I_lowGPU,1),2:3:6,3:3:6]
end

index1 = 1:mag:imSizeY*mag;
Yindex = index1;
for i = 2:mag
    Yindex(end+1:end+imSizeY) = index1 +(i-1);%[1:3:size(I_lowGPU,1),2:3:6,3:3:6]
end
inv_index1 = 1:imSizeY:imSizeY*mag;
inv_Yindex = inv_index1;
for i = 2:imSizeY
    inv_Yindex(end+1:end+mag) = inv_index1 +(i-1);%[1:3:size(I_lowGPU,1),2:3:6,3:3:6]
end

tic
for tt_z=1:size(z_d1,2)
    d1=z_d1(tt_z);
    d2=z_d2(tt_z);
    ObjectRecovery = imInitial_objpad;
    PatternRecovery = imInitial_ptpad;
%     clear imInitial_ptpad imInitial_objpad
    % subpixel shift parameters
    fy=ifftshift(gpuArray.linspace(-floor(imSizeY*mag/2),ceil(imSizeY*mag/2)-1,imSizeX*mag));
    fx=ifftshift(gpuArray.linspace(-floor(imSizeX*mag/2),ceil(imSizeX*mag/2)-1,imSizeY*mag));
    [FX,FY]=meshgrid(fx,fy);
    clear fx fy

    % prop parameters
    k0=2*pi/waveLength;
    kmax=pi/pixelSize;
    kxm0=gpuArray.linspace(-kmax,kmax,imSizeY*mag);
    kym0=gpuArray.linspace(-kmax,kmax,imSizeX*mag);
    [kxm,kym]=meshgrid(kxm0,kym0);
    kzm=single(sqrt(complex(k0^2-kxm.^2-kym.^2)));
    % H prop
    H_d1=gather(exp(1i.*d1.*real(kzm)).*exp(-abs(d1).*abs(imag(kzm))).*((k0^2-kxm.^2-kym.^2)>=0)); 
    inv_H_d1=gather(exp(1i.*(-d1).*real(kzm)).*exp(-abs((-d1)).*abs(imag(kzm))).*((k0^2-kxm.^2-kym.^2)>=0)); 
    H_d2=(exp(1i.*d2.*real(kzm)).*exp(-abs(d2).*abs(imag(kzm))).*((k0^2-kxm.^2-kym.^2)>=0)); 
    inv_H_d2=(exp(1i.*(-d2).*real(kzm)).*exp(-abs((-d2)).*abs(imag(kzm))).*((k0^2-kxm.^2-kym.^2)>=0)); 
    clear kxm0 kym0 kxm kym kzm
    
    % Prop to diffuser
    ObjectRecoveryProp=ifft2(ifftshift(H_d1.*fftshift(fft2(ObjectRecovery))));
    for loopnum=1:LoopN1
        for tt=1:imNum
        disp([loopnum tt]);
        
        % Shift wavefront
        Hs=exp(-1j*2*pi.*(FX.*-xlocation(imRandNum(tt),1)/imSizeX+FY.*-ylocation(imRandNum(tt),1)/imSizeY));
        Objectshift=ifft2(fft2(ObjectRecoveryProp).*Hs);
        clear Hs
        
        Pattern_plane = Objectshift.*PatternRecovery;   
        CCD_plane=ifft2(ifftshift(H_d2.*fftshift(fft2(Pattern_plane))));
        
        CCD_Intensity_kernal = conv2(abs(CCD_plane).^2, kernal);
        CCD_planeAmpDown = sqrt(CCD_Intensity_kernal(mag:mag:end,mag:mag:end));
        clear CCD_Intensity_kernal CCD_planeIntensity
        
%         I_lowGPU=(single(imSeq_t2_recover(:,:,imRandNum(tt)))).^0.7;
%         tt_Intensity=1;
%         if loopnum>0
%         tt_Intensity=mean(CCD_planeAmpDown(:))/mean(I_lowGPU(:));
%         end
%         I_lowGPU=tt_Intensity*I_lowGPU;

        I_lowGPU=(single(gpuArray(imSeq_t2_recover(:,:,imRandNum(tt))))).^0.7;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         %% m1
%         CCD_plane_new=gpuArray(zeros(imSizeX*mag,imSizeY*mag,'single'));
%         for iii=1:mag
%             for jjj=1:mag
%                 CCD_plane_new(iii:mag:end,jjj:mag:end)=(I_lowGPU(:,:)).*CCD_plane(iii:mag:end,jjj:mag:end)./CCD_planeAmpDown;
%             end
%         end
        
        % m2
        temp_amp_replace = (I_lowGPU)./(CCD_planeAmpDown);
        fun = @(block_struct) (block_struct.data).*temp_amp_replace;
        CCD_plane_new = blockproc(CCD_plane(Xindex,Yindex),[imSizeX imSizeY],fun);
        CCD_plane_new = CCD_plane_new(inv_Xindex,inv_Yindex);
        clear temp_amp_replace CCD_plane CCD_planeAmpDown I_lowGPU

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Pattern_plane_new=ifft2(ifftshift(inv_H_d2.*fftshift(fft2(CCD_plane_new))));
        clear CCD_plane_new

        Objectshift = Objectshift + gamaO*conj(PatternRecovery).*(Pattern_plane_new-Pattern_plane)./(alphaO.*max(max(abs(PatternRecovery).^2))+(1-alphaO).*(abs(PatternRecovery)).^2);
        PatternRecovery = PatternRecovery + gamaP*conj(Objectshift).*(Pattern_plane_new-Pattern_plane)./(alphaP.*max(max(abs(Objectshift).^2))+(1-alphaP).*(abs(Objectshift)).^2);
        clear Pattern_plane_new

        Hs=exp(-1j*2*pi.*(FX.*xlocation(imRandNum(tt),1)/imSizeX+FY.*ylocation(imRandNum(tt),1)/imSizeY));
        ObjectRecoveryProp=ifft2(fft2(Objectshift).*Hs);
        end 
    % Prop back
    ObjectRecovery=ifft2(ifftshift(inv_H_d1.*fftshift(fft2(ObjectRecoveryProp))));
    
    % Show result
%     figure('Name',[num2str(d1*1e6),'---',num2str(d2*1e6)]);
%     set(gcf,'outerposition',get(0,'ScreenSize'))
%     subplot(1,2,1);imshow(abs(ObjectRecovery(767*mag-255:767*mag+256,767*mag-255:767*mag+256)),[]); title([num2str(loopnum),' objAmp Recovery'])
%     subplot(1,2,2);imshow(angle(ObjectRecovery(767*mag-255:767*mag+256,767*mag-255:767*mag+256)),[]); title([num2str(loopnum),' objPhase Recovery'])
%     pause(0.1)     
    end
    save(['USAF_randomScan_pdot7_800_',num2str(d1*1e6),'_d2z',num2str(d2*1e6),'_',num2str(LoopN1),'loops_pad',num2str(mag),'_09182020.mat'],...
        'ObjectRecovery','PatternRecovery','d2','d1','-v7.3');
end
toc

%%
%mag=1;
% [phase_unwrap,~]=Unwrap_TIE_DCT_Iter(angle(gather(ObjectRecovery())));
[phase_unwrap,~]=Unwrap_TIE_DCT_Iter(angle(gather(ObjectRecovery(512-255:512+256,512-255:512+256))));
figure;
% imshow(angle(ObjectRecovery),[]); 
imshow(phase_unwrap,[-2 2]);
%subplot(1,2,2);
imshow(abs(PatternRecovery),[]);
%%
figure;
imshow(abs(ObjectRecovery),[]);
% pause(1)
%%
figure;imshow(abs(ObjectRecovery(512-255:512+256,512-255:512+256)),[])
%%
close all
for z0=-0e-6:2e-6:0e-6
    ObjectRecoveryShow=propTF_GPU(ObjectRecovery,pixelSize,waveLength,-z0);
%     ObjectRecoveryShow=propTF_GPU(ObjectRecovery(end/2-256:end/2+256,end/2-255:end/2+256),pixelSize,waveLength,-z0);
%    [phase_unwrap,~]=Unwrap_TIE_DCT_Iter(angle(gather(ObjectRecoveryShow)));
    figure;
   
    set(gcf,'outerposition',get(0,'screensize'));
    
    imshow(abs(ObjectRecoveryShow),[],'initialmagnification',1000);title(z0);
%     imshow((phase_unwrap),[],'initialmagnification',1000);title(z0);
end
% (1*mag:500*mag,1*mag:500*mag)

%%
for i =1:1521
    figure(100)
    imshow(abs(imSeq_t2(:,:,i)),[]);
    pause
end
