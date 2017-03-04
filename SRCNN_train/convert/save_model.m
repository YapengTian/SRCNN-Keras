%% save model / .mat
clc;
clear all;
close all;
model = {};

for k = 1 : 3
    strw = ['w',num2str(k-1),'.mat'];
    load(strw)
    model.weight{k} = array;
    strb = ['b',num2str(k-1),'.mat'];
    load(strb)
    model.bias{k} = array;
end
save srcnn_keras model;
