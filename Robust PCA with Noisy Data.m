%  Robust PCA with Noisy Data

%%
clear all; clc;
load('Image_anomaly.mat');

M = X;      % matrix, noisy data
S0 = randn(size(M));
tol = 1e-5;
err = 100;
lambda = 0.001;
factor = 0.01;
gamma = lambda/factor;
iter = 1;
maxiter= 10000;
L = zeros(size(M));
while (err > tol)&&(iter<maxiter),
    %disp(iter);   
    % Update L
    [U S V] = svd(M-S0);
    diagS = diag(S);
    svp = length(find(diagS>gamma/2));
    L = U(:,1:svp)*diag(diagS(1:svp)-gamma/2)*V(:,1:svp)';
    % Update S
    Sp = sign(M-L).*max(abs(M-L) -lambda/2,0); 
    err = norm(S0-Sp);
    S0=Sp; 
    iter = iter + 1;
end
figure; imagesc(M); title('Original'); saveas(gcf, 'figQ3original.png');
figure; imagesc(S0); title('Noise'); saveas(gcf, 'figQ3Noise.png');
figure; imagesc(L); title('De-noised'); saveas(gcf, 'figQ3Denoised.png');