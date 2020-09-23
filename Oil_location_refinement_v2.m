close all
clear
clc
%%
load('RandomScann_Target_1500_Eff_17_Del_500.mat')
% figure;imshow(imSeq_t2(1270:1269+512,1575:1574+512,1),[]);   
figure;imshow(imSeq_t2(:,:,1),[]);   
%%
% addpath(genpath(pwd))
% imSeq_t2
% imSeq_calpos=imSeq_t2(1100-511:1100+512,2150-511:2150+512,:);
imSeq_calpos=imSeq_t2(1600+1:1600+512,3300+1:3300+512,:);
% imSeq_calpos=imSeq(1500-255:1500+256,1+50:512+50,:);

figure;imshow(imSeq_calpos(:,:,1),[]);
%%
imSizeX = size(imSeq_calpos,1); %image size
imSizeY = size(imSeq_calpos,2); %image size
imNum = size(imSeq_calpos,3); %image number
waveLength = 0.532e-6; %waveLength
mag = 1; 
pixelSize0 = 1.85e-6; 
pixelSize = pixelSize0/mag; %pixelSize
startImg=1;endImg=imNum;
showStep=1;
pos1_X_dftpc=zeros(floor((endImg-startImg)/showStep),1);
pos1_Y_dftpc=zeros(floor((endImg-startImg)/showStep),1);
tt=0;
usfac=100;
standardImg=gather(single(imSeq_calpos(:,:,1)));
for i=startImg:showStep:endImg
    tt=tt+1;
    disp(tt);
    copyImg=single(imSeq_calpos(:,:,i));
    [output, Greg] = dftregistration(fft2(standardImg),fft2(copyImg),usfac);
    pos1_Y_dftpc(tt,1) = output(3);
    pos1_X_dftpc(tt,1) = output(4);
end
xlocation=pos1_X_dftpc;
ylocation=pos1_Y_dftpc;

figure(1);
plot(xlocation,ylocation,'r*');
%%
save(['loc_dftpc_XY.mat'],'xlocation','ylocation');
%%
% imSeq_calpos=imSeq_t2(1100-255:1100+256,2030-255:2030+256,:);
% imSeq_calpos=imSeq_t2(1700-255:1700+256,1+20:512+20,:);
imSeq_calpos=imSeq_t2(1010:1009+512,3235:3234+512,:);
% imSeq_calpos=imSeq_t2(1400-255:1400+256,1+50:512+50,:);
% imSeq_calpos=imSeq_t2(1100-511:1100+512,2150-511:2150+512,:);
% imSeq_calpos=imSeq(1150-255:1150+256,2130-255:2130+256,:);
imSizeX = size(imSeq_calpos,1); %image size
imSizeY = size(imSeq_calpos,2); %image size
imNum = size(imSeq_calpos,3); %image number
imInitial_obj=zeros(imSizeX,imSizeY,'single');  
for i=1:size(imSeq_calpos,3)
    disp(i);
    imInitial_obj=imInitial_obj+abs(subpixelshift3GPU((single(imSeq_calpos(:,:,i))),xlocation(i,1),ylocation(i,1))); %shift back
end
imInitial_obj=imInitial_obj/imNum;
imInitial_obj=gpuArray(imInitial_obj);
imInitial_pad=sqrt(single(mean(imSeq_calpos,3)));

% figure;
% subplot(121);imshow(abs(imInitial_obj),[]);title('initial object pattern'); % show initial guess
% subplot(122);imshow(abs(imInitial_pad),[]);title('initial padding pattern');
%
startImg=1;endImg=imNum;
showStep=1;
pos1_X_dftpc=zeros(floor((endImg-startImg)/showStep),1);
pos1_Y_dftpc=zeros(floor((endImg-startImg)/showStep),1);
tt=0;
usfac=100;

standardImg=gather(imInitial_obj);
for i=startImg:showStep:endImg
    tt=tt+1;
    disp(tt);
    copyImg=single(imSeq_calpos(:,:,i));
    [output, Greg] = dftregistration(fft2(standardImg),fft2(copyImg),usfac);
    pos1_Y_dftpc(tt,1) = output(3);
    pos1_X_dftpc(tt,1) = output(4);
end
xlocation=pos1_X_dftpc;
ylocation=pos1_Y_dftpc;

figure;
plot(xlocation,ylocation,'r*');
