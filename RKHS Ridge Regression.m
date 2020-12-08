% RKHS Ridge Regression

%% 

clear all; clc;
load('peaks.mat');

%% Construct Gram matrix

bwd = 1; 
n_points = 100;

figure; imagesc(Y); title('Original');
saveas(gcf, 'figQ4Original.png');

lambda = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
std_err= zeros(1,length(lambda));

for i = 1:length(lambda),
    [img_ridge, err_ridge, MSE_ridge] = ...
        kernel_ridge_reg(lambda(i),n_points,bwd,Y);
    figure; imagesc(img_ridge);
    title(['Lambda = ', num2str(lambda(i))]);
    figname = ['figQ4Lambda',num2str(i),'.png']
    saveas(gcf,figname);    
    std_err(i) = std(err_ridge);
end

function [img_ridge, err_ridge, MSE_ridge] = ...
    kernel_ridge_reg(lambda, n_points,bwd,Y),
% Reconstruct 2D image using Kernel ridge regression
% Input:
%       lambda:     smoothing parameter lambda
%       n_points:   number of points in Kernel
%       bwd:        Kernel bandwidth
%       Y:          2D original image

    x = linspace(1,n_points, n_points);    
    gauss1d = exp(-dist(x).^2/bwd);
    K = kron(gauss1d,gauss1d);
    Y_vector = reshape(Y,n_points*n_points,1);
    Kridge = K + lambda*eye(size(K,1));
    alpha = Y_vector\Kridge;
    img_ridge = K*alpha';
    err_ridge = Y_vector - img_ridge;
    MSE_ridge = mean(err_ridge.^2);
    img_ridge = reshape(img_ridge, size(Y));  
end


