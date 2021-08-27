% QUESTIONS:
% 1. how to call the function score?
% 2. is the error score(X, whatever output)? 
% 3. number of factors vs rank (r) -> kolda's paper

% normalization method 1: 
faces_norm1 = importdata('faces_norm1.mat');
faces_norm1 = reshape(faces_norm1,[1000 96 96]);
X = tensor(faces_norm1);

%% 
% 1. alternating least squares for CP decomposition:

% create random initial guess
rng('default'); 

maxR = 30;
errors_als_rand_init = zeros([maxR 1]);
norm_X = norm(X);

for r = 1:maxR
    M_als_rand_init = cp_als(X, r, 'tol',1e-6, 'maxiters',100, 'printitn', 0);
    
    Y = X - tensor(M_als_rand_init);
    errors_als_rand_init(r) = norm(Y) / norm_X;
    
    % ./ is element-wise operation
    % close all to close all figures
    % ctrl + enter is run section
end 

%%
% nvecs as initial guess

errors_als_nvecs_init = zeros([maxR 1]);

for r = 1:maxR
    M_als_nvecs_init = cp_als(X, r, 'init','nvecs', 'tol',1e-6, 'maxiters',100, 'printitn', 0);
    Y = X - tensor(M_als_nvecs_init);
    errors_als_nvecs_init(r) = norm(Y) / norm_X;
    
end 

%%

% 2. all-at-once optimization for CP decomposition:

% without the non-negative constraint:
errors_opt = zeros([maxR 1]);
maxR = 15;

for r = 1:maxR
    M_opt_init = create_guess('Data', X, 'Num_Factors', r, 'Factor_Generator', 'nvecs');
    [M_opt, M0_opt, output_opt] = cp_opt(X, r, 'init', M_opt_init);
    
    Y = X - tensor(M_opt);
    errors_opt(r) = norm(Y) / norm_X;
end 
errors_opt = errors_opt(1:15);

%% with the non-negative constraint:
errors_opt_nonneg = zeros([maxR 1]);

for r = 1:maxR
 % with the non-negative constraint:
    M_opt_init_nonneg = create_guess('Data', X, 'Num_Factors', r, 'Factor_Generator', 'nvecs');
    [M_opt_nonneg, M0_opt_nonneg, output_opt_nonneg] = cp_opt(X, r, 'init', M_opt_init_nonneg, 'lower', 0);
    
    Y = X - tensor(M_opt_nonneg);
    errors_opt_nonneg(r) = norm(Y) / norm_X;
end

%% Plot and compare the reconstruction curves

figure; hold on

a1 = plot(errors_als_rand_init, 'color', 'b');

a2 = plot(errors_als_nvecs_init, 'color', 'g');

a3 = plot(errors_opt, 'color', 'm');

a4 = plot(errors_opt_nonneg, 'color', 'r');

legend([a1;a2;a3;a4], 'ALS with random initial guess','ALS with nvecs initial guess', 'Regular Direct Optimization', 'Nonnegative Direct Optimization');

title("Reconstruction error curves (using np.max normalization)");

%% Visualize the factors as images:

maxR = 5;

figure; hold on

for r = 1:maxR
    f = M_als_rand_init.u{2}(:,r) * M_als_rand_init.u{3}(:,r)';
    subplot(4, maxR, r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
end

for r = 1:maxR
    f = M_als_nvecs_init.u{2}(:,r) * M_als_nvecs_init.u{3}(:,r)';
    subplot(4, maxR, maxR + r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
end

for r = 1:maxR
    f = M_opt.u{2}(:,r) * M_opt.u{3}(:,r)';
    subplot(4, maxR, maxR*2 + r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
end

for r = 1:maxR
    f = M_opt_nonneg.u{2}(:,r) * M_opt_nonneg.u{3}(:,r)';
    subplot(4, maxR, maxR*3 + r);
    imagesc(f');
    % matlab reads by columns -> need to transpose
    axis image;
    % preserve image aspect ratio
end