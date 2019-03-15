%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               Growth & Development - Take-Home Exam                     %
%                                                                         %
%                           Gabriela Barbosa                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; 
clc; 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        QUESTION 1 - Factor Input Misallocation in the Village           % 
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

seed = rng;
%rng(seed);

%%% Variables moments (in logs)
smean=1;
svar = 1.416;
kmean=1;
kvar=0.749;
skcor = 0; % For QUESTION 1.6: change it for 0.25 
skcov = skcor*sqrt(svar)*sqrt(kvar);

mu = [smean kmean];
varcov = [svar skcov; skcov kvar];

%%% Simulation

% In logs:
S = mvnrnd(mu,varcov,10000);
%lns = S(:,1);
%lnk = S(:,2);

% In level:
Sl=exp(S);
s = Sl(:,1);
k = Sl(:,2);

%%% Generated data

adm_data_nocor = [s k]; % For latter use

%%% Summary statistics
mean(s)
summary(dataset(s));
var(s)

mean(k)
summary(dataset(k));
var(k)
                   
%%
%%% Plot in logs

x1 = S(:,1); 
x2 = S(:,2);
x = linspace(min(x1),max(x1),100); % x axis
y =linspace(min(x2),max(x2),100); % y axis
[X Y] = meshgrid(x,y); % all combinations of x, y
Z = mvnpdf([X(:) Y(:)],mu,varcov); % compute Gaussian pdf
Z = reshape(Z,size(X)); % put into same size as X, Y

figure(1)
surf(X,Y,Z)
xlabel('ln s'); ylabel('ln k'); zlabel('pdf');
title({'{\bf\fontsize{14} Joint Probability Density}'; 'in logs and \rho_{sk}=0'},'FontWeight','Normal')

saveas(gcf, 'pdf_16.png')

%%% Plot in levels

x1l = Sl(:,1); 
x2l = Sl(:,2);
xl = linspace(min(x1l),max(x1l),100);
yl =linspace(min(x2l),max(x2l),100); 
[Xl Yl] = meshgrid(xl,yl);
Zl=exp(Z);
Zl = reshape(Zl,size(Xl));

figure(2)
surf(Xl,Yl,Zl)
xlabel('s'); ylabel('k'); zlabel('pdf');
title({'{\bf\fontsize{14} Joint Probability Density}'; 'in levels and \rho_{sk}=0'},'FontWeight','Normal')

saveas(gcf, 'pdf2_16.png')

clear x x1 x1l x2 x2l xl Xl Xl y Y yl Yl Z Zl

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Parameter

gama = 0.8; % For QUESTION 2: change it for 0.8

%%% Output

y=s.*k.^gama;

mean(y)
summary(dataset(y));
var(y)

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Since we have 10,000 observations, there are 10,000 individuals/agents.
%   As derived in class, we can write the Planner's problem isolating one
% random agents - say i=1.
%   The Planner's optimality conditions:
%       zi=si^(1/(1-gama))
%       zi k1=z1 ki
%       k1 = (z1/Z) K

%%% Define the following: Write in terms of the first agents i=1 

z=s.^(1/(1-gama));
s1=s(1);
z1=z(1);

%%% Aggregate variables

K=sum(k);
Z=sum(z);

%%% Optimal conditions (FOCs)

% For the first agent
k1 = (z1/Z)*K;

% For all others
ke= ( (s1*k1.^(gama-1))./s ).^(1/(gama-1));

%%% Checking marginal productivities

mp = s.*ke.^(gama-1);

if range(mp) <  1e-10
    disp('This is indeed the efficient allocation: all marginal products are equalized!');
else
    disp('Something is wrong: efficient allocation not found!')
end

clear k1 z1 Z s1

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   To compare capital data and its efficient allocation, I am going to 
% check their distribution, summary statistics, marginal products and 
% the value difference between these allocations.

%%% Summary Statistics (notice that their mean is the same, since there 
% is just redistribution going around)

summary(dataset(k));
summary(dataset(ke));
mean(k)

if (mean(k)- mean(ke) < 1e-10)
    disp('Efficient allocation implies redistribution of capital! ')
else
    disp('Something is wrong: check code!')
end

var(k)
var(ke)

%%% Difference

mean(k-ke)
summary(dataset(k-ke));
var(k-ke)

%%% Capital Distribution

figure(3)
histogram(log(k), 'EdgeColor', 'green', 'FaceColor',  'green', 'FaceAlpha', 0.2)
hold on
histogram(log(ke),'EdgeColor', 'blue', 'FaceColor',  'blue','FaceAlpha', 0.2 )
title({'{\bf\fontsize{14} ( Log ) Capital Distribution - Data vs. Social Optimal}'; '\gamma=0.8, \rho_{sk}=0.25'},'FontWeight','Normal')
ylabel('log k')
xlabel('number on individuals')
legend('Sample (Log) Capital', 'Efficient (Log) Capital' )
saveas(gcf, 'k_246.png')

figure(4)
scatter(log(s),log(k))
hold on
scatter(log(s),log(ke))
title({'{\bf\fontsize{14} Capital (Mis)Allocation}'; '\gamma=0.8, \rho_{sk}=0.25'},'FontWeight','Normal')
ylabel('log k')
xlabel('log s')
legend('Data', 'Efficient' )
saveas(gcf, 'k2_246.png')

%%% Marginal Products

mp_ne = s.*k.^(gama-1);

figure(5)
scatter(log(s),log(mp_ne))
hold on
scatter(log(s),log(mp))
title({'{\bf\fontsize{14} Marginal Product of Capital}'; '\gamma=0.8, \rho_{sk}=0.25'},'FontWeight','Normal')
ylabel('log MPK')
xlabel('log s')
legend('Data', 'Efficient' )
saveas(gcf, 'k3_246.png')

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Efficient output

ye = s.*ke.^gama;
Ye = sum(ye);

%%% Ouptut gains from reallocation

Y = sum(y);
gain = Ye/Y;
log(gain);
gain_i = ye./y; % Individuals' gains

mean(gain_i)
summary(dataset(gain_i));
var(gain_i)

h=histfit(log(gain_i));
set(h(1),'facecolor','b','EdgeColor', 'b','FaceAlpha', 0.2); set(h(2),'color','r')
title({'{\bf\fontsize{14} ( Log ) Individual Output Gain}'; '\gamma=0.8, \rho_{sk}=0.25'},'FontWeight','Normal')
ylabel('log gain')
xlabel('number on individuals')
saveas(gcf, 'gain_256.png')

clear h 

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Now we assume correlation between ln(si) and ln(ki) is 0.25 and re-do 
% the above.
%   For the sake of simplicity, I will not copy/paste the above code here
% since the proceedure is exactly the same apart from the change in the
% correlation parameter value.
%   Hence, to do the present question one need just to change the 
% correlation value in the code. I explicity specify where this change 
% should be made above.
%   Speficially, the change is on LINE 25.
%   Resulting data set is stored under adm_data_cor (for latter use).
%   Please refer to the solution document for the results for this section.


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    QUESTION 2 - Higher Span Control                     % 
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   For the sake of simplicity, I will not copy/paste the code above here.
%   The proceedure in this question is exactly the same as above but the 
% only difference is for the parameter gama: now is 0.8.
%   Please refer to the solution document for the set of results.
%   If you wish to run this question as well, it is pointed out in the
% above code where you should make a change (precisely, in LINE 98).


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   QUESTION 3 - From Administrative to Random Sampling in the Village    % 
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Random sample 10 observations, 100 times, from the data set generated 
% on question 1.1

%adm_data_nocor = [s k]; $ Data with no correlation between s and k
sample_s = NaN(10,100);
sample_k = NaN(10,100);

for i = 1:100
    sample_s(:,i) = randsample(adm_data_nocor(:,1),10); 
    sample_k(:,i) = randsample(adm_data_nocor(:,2),10); 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Parameter

gama = 0.5;

%%% Resulting matrices

data_y = NaN(10,100);
data_ke = NaN(10,100);
data_mp = NaN(10,100);
data_ye = NaN(10,100);
data_gain = NaN(1,100);
data_gain_ind = NaN(10,100);

%%% Loop: computation for each of the 100 samples

for i = 1:100
    
    %%% Specifying the sample to use as data
    data_s = sample_s(:,i);
    data_k = sample_k(:,i);
    
    %%% Output
    data_y(:,i) = data_s.*data_k.^gama;
    
    %%% Define the following: Write in terms of the first agents i=1 
    data_z = data_s.^(1/(1-gama));
    data_s1 = data_s(1);
    data_z1 = data_z(1);

    %%% Aggregate variables
    data_K = sum(data_k);
    data_Z = sum(data_z);

    %%% Optimal conditions (FOCs)

    % For the first agent
    data_k1 = (data_z1/data_Z)*data_K;

    % For all others
    data_ke(:,i) = ( (data_s1*data_k1.^(gama-1))./data_s ).^(1/(gama-1));

    %%% Checking marginal productivities
    data_mp(:,i) = data_s.*data_ke(:,i).^(gama-1);

    if range(data_mp(:,i)) >  1e-10
        disp('Something is wrong: efficient allocation not found!')
    end
    
    %%% Efficient output
    data_ye(:,i) = data_s.*data_ke(:,i).^gama;
    data_Ye = sum(data_ye(:,i));

    %%% Ouptut gains from reallocation
    data_Y = sum(data_y(:,i));

    data_gain(1,i) = data_Ye/data_Y;

    %%% Individual's gains:
    data_gain_ind(:,i) = data_ye(:,i)./data_y(:,i);

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   To compare the distribution of output gains from reallocation from the 
% 100 random samples with the administrative data

histogram(data_gain, 'EdgeColor', 'blue', 'FaceColor',  'blue', 'FaceAlpha', 0.2)
hold on
line([1.9697, 1.9697], ylim, 'LineWidth', 2, 'Color', 'r');
title({'{\bf\fontsize{14} Distribution of Output Gains - 100 random samples}'; '\gamma=0.5, \rho_{sk}=0'},'FontWeight','Normal')
ylabel('number of samples')
xlabel('output gains')
legend('Samples Distribution', 'Adm. Data level' )
saveas(gcf, 'k3_3.png')

mean(data_gain)
summary(dataset(data_gain'))
var(data_gain)

clear data_gain data_gain_ind data_k data_K data_k1 data_ke data_mp ...
    data_s data_s1 data_y data_Y data_ye data_Ye data_z data_Z data_z1 
   

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   QUESTION 4 - Endogenous Productivity                  % 
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   The data set is the generated sample with a 0 correlation between
% ln(si) and ln(ki) from question 1.1.
%   I will assume a is actually s in question 1. Then, the data set 
% consists of observations for ln(ai) and ln(ki).
%   With these variables then I am able to generate the managerial ability 
% si consisting of true ability (ai) and capital(ki)

a = adm_data_nocor(:,1);
k = adm_data_nocor(:,2);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Given the CES specification of managerial ability si and the assumption
% that sigma = 1 (elasticity of substitution), this is actually a Cobb
% Douglas (production) function.

%%% Parameters

gama = 0.5; 
alfa = 0.5; % degree of complementary between a and k
sigma = 2;  % QUESTION 5: 0.5, 2.0
            % elasticity of substitution

es = (sigma - 1)/sigma;

%%% Managerial ability (si) function:

if sigma == 1
    s = a.^alfa.*k.^(1-alfa);
else
    s = ( alfa.*a.^es + (1-alfa).*k.^es ).^(1/es);
end

y = s.*k.^gama;

data = [s k y];

%%% Summary statistics

summary(dataset(s))
mean(s)
var(s)

summary(dataset(y))
mean(y)
var(y)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Since I am using si data generated above, we can write the Planner's
% problem exactly like before. That is, 
%       Ye = max_ki sum_i si ki^gamma 


%%% Define the following: Write in terms of the first agents i=1 

z = s.^(1/(1-gama));
s1 = s(1);
z1 = z(1);

%%% Aggregate variables

K = sum(k);
Z = sum(z);

%%% Optimal conditions (FOCs)

% For the first agent
k1 = (z1/Z)*K;

% For all others
ke = ( (s1*k1.^(gama-1))./s ).^(1/(gama-1));

%%% Checking marginal productivities

mp = s.*ke.^(gama-1);

if range(mp) <  1e-10
    disp('This is indeed the efficient allocation: all marginal products are equalized!');
else
    disp('Something is wrong: efficient allocation not found!')
end

clear k1 z1 Z s1

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

summary(dataset(k));
summary(dataset(ke));
mean(k)

if (mean(k)- mean(ke) < 1e-10)
    disp('Efficient allocation implies redistribution of capital! ')
else
    disp('Something is wrong: check code!')
end

var(k)
var(ke)

%%% Difference

mean(k-ke)
summary(dataset(k-ke));
var(k-ke)

%%% Capital Distribution

figure(3)
histogram(log(k), 'EdgeColor', 'green', 'FaceColor',  'green', 'FaceAlpha', 0.2)
hold on
histogram(log(ke),'EdgeColor', 'blue', 'FaceColor',  'blue','FaceAlpha', 0.2 )
title({'{\bf\fontsize{14} ( Log ) Capital Distribution - Data vs. Social Optimal}'; '\gamma=0.5, \rho_{sk}=0, \alpha=0.5, \sigma=2'},'FontWeight','Normal')
ylabel('log k')
xlabel('number on individuals')
legend('Sample (Log) Capital', 'Efficient (Log) Capital' )
saveas(gcf, 'k_47.png')

figure(4)
scatter(log(s),log(k))
hold on
scatter(log(s),log(ke))
title({'{\bf\fontsize{14} Capital (Mis)Allocation}'; '\gamma=0.5, \rho_{sk}=0, \alpha=0.5, \sigma=2'},'FontWeight','Normal')
ylabel('log k')
xlabel('log s')
legend('Data', 'Efficient' )
saveas(gcf, 'k2_48.png')

%%% Marginal Products

mp_ne = s.*k.^(gama-1);

figure(5)
scatter(log(s),log(mp_ne))
hold on
scatter(log(s),log(mp))
title({'{\bf\fontsize{14} Marginal Product of Capital}'; '\gamma=0.5, \rho_{sk}=0, \alpha=0.5, \sigma=2'},'FontWeight','Normal')
ylabel('log MPK')
xlabel('log s')
legend('Data', 'Efficient' )
saveas(gcf, 'k3_49.png')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Efficient output

ye = s.*ke.^gama;
Ye = sum(ye);

%%% Ouptut gains from reallocation

Y = sum(y);

gain = Ye/Y;
log(gain)

%%% Individual's gains:

gain_i = ye./y;

mean(gain_i)
summary(dataset(gain_i));
var(gain_i)

h=histfit(log(gain_i));
set(h(1),'facecolor','b','EdgeColor', 'b','FaceAlpha', 0.2); set(h(2),'color','r')
title({'{\bf\fontsize{14} ( Log ) Individual Output Gain}'; '\gamma=0.5, \rho_{sk}=0, \alpha=0.5, \sigma=2'},'FontWeight','Normal')
ylabel('log gain')
xlabel('number on individuals')
saveas(gcf, 'gain_43.png')

clear h 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Now the only change made is on the elasticity of substitution between
% a and k parameter.
% There are two cases: sigma = 0.5 and 2.0.
% For the sake of simplicity I will not copy/paste the code above here. One
% should simply change this parameter value in the above specification. It
% is explicity pointed out where this change should be made. It is in LINE
% 408.
% Please refer to the solution document for the results.















