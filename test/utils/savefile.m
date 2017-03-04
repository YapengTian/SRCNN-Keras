function [] = savefile(imLR, ori_HR, im_rgb,imbicubic, filename)
    
    filename = filename(1:end-4);
%     imLR = imresize( ori_HR, 1.0/3, 'Bicubic');
    imwrite(uint8(imLR),  ['Result\', filename, '_LR.bmp']);
    imwrite(uint8(ori_HR),  ['Result\', filename, '_HR.bmp']);
    imwrite(uint8(im_rgb),  ['Result\', filename, '_srcnn.bmp']);
    imwrite(uint8(imbicubic),  ['Result\', filename, '_bicubic.bmp']);
end