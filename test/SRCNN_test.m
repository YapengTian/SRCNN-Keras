close all;clear all; clc;
run matconvnet/matlab/vl_setupnn;

addpath('utils')
load('SRCNN_keras.mat')
% set parameters
up_scale = 3;
%%
im_path = 'data/Set5/';
im_dir = dir( fullfile(im_path, '*bmp') );
im_num = length( im_dir );
for img = 1:im_num,
X = imread( fullfile(im_path, im_dir(img).name) );
grd = X;
if size(X,3) == 3
    X = rgb2ycbcr(X);
    X = double(X(:,:, 1));
else
    X = double(X);
end
X = modcrop(X, up_scale);
grd = modcrop(grd, up_scale);
X = double(X);
[row, col, ~] = size(X);
%Generate LR image
im_l = imresize(X, 1/up_scale, 'bicubic')/255;
%% SRCNN
im_h_y = SRCNN_Matconvnet(im_l, model, up_scale);
im_h = double(im_h_y * 255);
%% Show
lr = imresize(grd, 1/up_scale, 'bicubic');
if size(lr, 3) == 3
    lr = rgb2ycbcr(lr);
    xy = uint8(im_l*255);
    xcb = lr(:, :, 2);
    xcr = lr(:, :, 3);
    lr(:,:, 1) = xy;
    bic(:, :, 1) = imresize(xy, up_scale, 'bicubic');
    bic(:, :, 2) = imresize(xcb, up_scale, 'bicubic');
    bic(:, :, 3) = imresize(xcr, up_scale, 'bicubic');
    im_bic = ycbcr2rgb(bic);
    bic(:, :, 1) = uint8(im_h);
    our = ycbcr2rgb(bic);
    lr = ycbcr2rgb(lr);
else
    im_bic = imresize(lr, up_scale, 'bicubic');
    our = uint8(im_h);
end
clear bic;

grd = shave(grd, [up_scale, up_scale]);
our = shave(our, [up_scale, up_scale]);
im_bic = shave(im_bic, [up_scale, up_scale]);
%% 
savefile( lr, grd, our,im_bic, im_dir(img).name);
%% Evaluation
up_scale = 3;
X = shave(uint8(X), [up_scale, up_scale]);
im_h = shave(uint8(im_h), [up_scale, up_scale]);
pp_psnr = compute_rmse(X, im_h);
scores(img, 1) = pp_psnr;
scores(img, 2) = ssim(X, im_h);
end
save Result/scores scores;
mean(scores)


