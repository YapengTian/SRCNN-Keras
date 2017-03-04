function im_h_y = SRCNN_Matconvnet(im_l_y,model, scale)

weight = model.weight;
bias = model.bias;
im_y = single(imresize(im_l_y,scale,'bicubic'));
convfea = vl_nnconv(im_y,weight{1},bias{1}, 'Pad',4);
convfea = vl_nnrelu(convfea);
convfea = vl_nnconv(convfea,weight{2},bias{2});
convfea = vl_nnrelu(convfea);
convfea = vl_nnconv(convfea,weight{3},bias{3}, 'Pad', 2);
im_h_y = convfea;

end