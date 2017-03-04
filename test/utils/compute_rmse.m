function [rmse] = compute_rmse(im1, im2)

if size(im1, 3) == 3,
    im1 = rgb2ycbcr(im1);
    im1 = im1(:, :, 1);
  
end

if size(im2, 3) == 3,
    im2 = rgb2ycbcr(im2);
    im2 = im2(:, :, 1);
  
end
yh = double(im1);
yl = double(im2);
rmse=10*log10(255.^2/mean((yl(:)-yh(:)).^2));
end