% QUESTIONS:
% 1. how to call the function score?
% 2. is the error score(X, whatever output)? 
% 3. number of factors vs rank (r) -> kolda's paper

data_faces = importdata('faces.mat');
data_faces = double(data_faces);
data_faces = reshape(data_faces,[1000 96 96]);
X = tensor(data_faces)

% 1. alternating least squares for CP decomposition:

% random initial guess
rng('default'); 

M_als_rand_init = cp_als(X, 3, 'tol',1e-6, 'maxiters',100, 'printitn',10);

errors_als_rand_init = zeros(15);

for r = 1:15
    M_als_rand_init = cp_als(X, r, 'tol',1e-6, 'maxiters',100, 'printitn', 0);
    errors_als_rand_init(r) = 1 - score(X, M_als_rand_init); 
end 

plot(errors_als_rand_init);

% nvecs as initial guess
M_als_nvecs_init = cp_als(X, 3, 'init','nvecs', 'tol',1e-6, 'maxiters',100, 'printitn',10);

errors_als_nvecs_init = zeros(15);

for r = 1:15
    M_als_nvecs_init = cp_als(X, r, 'tol',1e-6, 'maxiters',100, 'printitn', 0);
    errors_als_nvecs_init(r) = 1 - score(X, M_als_nvecs_init); 
end 

plot(errors_als_nvecs_init);

% 2. all-at-once optimization for CP decomposition:

R = 5; 

M_opt_init = create_guess('Data', X, 'Num_Factors', R, 'Factor_Generator', 'nvecs');

% without the non-negative constraint:
[M_opt, M0_opt, output_opt] = cp_opt(X, R, 'init', M_opt_init);
exitmsg = output_opt.ExitMsg;
fit = output_opt.Fit;

errors_opt = zeros(15);

for r = 1:15
    [M_opt, M0_opt, output_opt] = cp_opt(X, r, 'init', M_opt_init);
    exitmsg = output_opt.ExitMsg;
    fit = output_opt.Fit;
    errors_opt(r) = 1 - score(X, M_opt); 
end 

plot(errors_opt);

% with the non-negative constraint:
[M_opt_nonneg, M0_opt_nonneg, output_opt_nonneg] = cp_opt(X, R, 'init', M_opt_init, 'lower', 0);
exitmsg_nonneg = output_opt_nonneg.ExitMsg;
fit_nonneg = output_opt_nonneg.Fit;

errors_opt_nonneg = zeros(15);

for r = 1:15
    [M_opt_nonneg, M0_opt, output_opt] = cp_opt(X, r, 'init', M_opt_init, 'lower', 0);
    exitmsg = output_opt.ExitMsg;
    fit = output_opt.Fit;
    errors_opt_nonneg(r) = 1 - score(X, M_opt_nonneg); 
end 

plot(errors_opt_nonneg);


