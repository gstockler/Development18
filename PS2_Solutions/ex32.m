%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Development - PS2                                    %
%                                                       %
%  Gabriela Barbosa                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; 
clc; 

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       QUESTION 3 - Welfare Costs of Seasons           %
%           Adding Seasonal Labor Supply                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Now we add labor supply to the agent's problem.
%   In term of welfare analysis, we can decompose the welfare
% effects into changes in consumption and changes in labor 
% allocations.
%   The setup is similar to the previous questions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%          Welfare gains of no idiosyncratic risk
%          C,H highly correlated

% 0) Set up

N = 1000; % number of individuals
T = 40; % number of periods (years) individual lives
M=12; % number of months in each period
ages=linspace(16,55,T); % ages throughout life

%%% Parameters

eta = 1; % coef. relative risk aversion
beta=.99; % discount factor
nu = 1; % Frisch elasticity 
kappa = .66*(1/.25)*(2/(28.5 * 30/7)^2); %  relative disutility weight of work

% idiosyncratic nonseasonal stochastic component; per period
sigma_eps=0.2;
log_eps = normrnd(0,sigma_eps,1,T);

%%% Consumption process: 

% initial permanent consumption level variance; at the beggining of life
sigma_u=0.2;

% initial permanent labor level variance; at the beggining of life
sigma_uh=0.2;

% deterministic seasonal component of consumption; common among agents
g=NaN(12,3); % three levels: low, mid and high
g(:,2)=[.863; .691; 1.151; 1.140; 1.094; 1.060; 1.037; 1.037; 1.037; 1.002; .968; .921  ]; 
g(:,1)=[.727; .381; 1.303; 1.280; 1.188; 1.119; 1.073; 1.073; 1.073; 1.004; .935; .843 ];
g(:,3)=[.932; .845; 1.076; 1.070; 1.047; 1.030; 1.018; 1.018; 1.018; 1.001; .984; .961 ];

% stochastic seasonal component; common among agents
sigma_m=NaN(12,3); % three levels: low, mid and high
sigma_m(:,2)=[.085; .068; .29; .283; .273; .273; .239; .205; .188; .188; .171; .137 ]; 
sigma_m(:,3)=[.171; .137; .580; .567; .546; .546; .478; .410; .376; .376; .341; .273 ];
sigma_m(:,1)=[.043; .034; .145; .142; .137; .137; .119; .102; .094; .094; .085; .068 ];

%%% Labor process: assuming higly positive correlation with its
%                  consumption counterpat

% initial permanent consumption level variance; at the beggining of life
% assume is the same as consumption

% deterministic seasonal component of consumption; common among agents
gh = g;

% stochastic seasonal component; common among agents
sigma_mh = sigma_m;

% SEASON CHOICE - Consumption
% deterministic
s=1; %1,2,3: low, med, high

% stochastic
ss=1; %1,2,3: low, med, high

% SEASON CHOICE - Labor
% deterministic
sh=3; %1,2,3: low, med, high

% stochastic
ssh=3; %1,2,3: low, med, high

%% 1) Welfare of C & H (det + stoc season) + risk

%%% Consumption & Labor loop
z=NaN(N,1);
c=NaN(M,T,N);

zh=NaN(N,1);
h=NaN(M,T,N);

for n = 1:N
    
    log_u = normrnd(0,sigma_u,1); 
    z(n,1) = exp(-sigma_u./2)*exp(log_u);

    log_uh = log_u; 
    zh(n,1) = z(n,1);
    
    for t = 1:T
        for m = 1:M
            
            %%% CONSUMPTION:
            % CHOICE for stochastic season: sigma_m(:,L/M/H)
            log_eps_m = normrnd(0,sigma_m(m,ss),1); %m=1/low; 2/mid; 3/high
            
            % CHOICE for deterministic season: g(:,L/M/H)
            c(m,t,n) = z(n,1)*g(m,s)*exp(-sigma_m(m,ss)./2)*exp(log_eps_m)*exp(-sigma_eps./2)*exp(log_eps(t));
            
                        
            %%% LABOR:
            % CHOICE for stochastic season: sigma_m(:,L/M/H)
            log_eps_mh = normrnd(0,sigma_mh(m,ssh),1); %m=1/low; 2/mid; 3/high
            
            % CHOICE for deterministic season: g(:,L/M/H)
            h(m,t,n) = zh(n,1)*gh(m,sh)*exp(-sigma_mh(m,ssh)./2)*exp(log_eps_mh)*exp(-sigma_eps./2)*exp(log_eps(t));
        
        end
    end
end

%%% Lifetime utility loop
W_c = NaN(N,1);
W_h = NaN(N,1);

for n = 1:N
    
    Wm_c = NaN(M,1);
    Wt_c = NaN(T,1);
    
    Wm_h = NaN(M,1);
    Wt_h = NaN(T,1);
    
    for t = 1:T
        for m = 1:M
                
               Wm_c(m,1) = beta.^(m-1) * log(c(m,t,n)); 
               Wm_h(m,1) = beta.^(m-1) * kappa * h(m,t,n).^(1+1/nu)./(1+1/nu); 
        
        end
        
        Wt_c(t,1) = beta.^(12*(t)) * sum(Wm_c,1);
        Wt_h(t,1) = beta.^(12*(t)) * sum(Wm_h,1);
        
    end
    
    W_c(n,1) = sum(Wt_c,1);
    W_h(n,1) = sum(Wt_h,1);
    
end

W =W_c-W_h;
%% 2) Welfare of C&H (det + stoc season) + NO risk

%%% Consumption
c_ns=NaN(M,T,N);
h_ns=NaN(M,T,N);

for n = 1:N
    for t = 1:T
        for m = 1:M
            
            c_ns(m,t,n) = c(m,t,n)./(exp(-sigma_eps./2)*exp(log_eps(t)));
            h_ns(m,t,n) = h(m,t,n)./(exp(-sigma_eps./2)*exp(log_eps(t)));
        
        end
    end
end


%%% Lifetime utility
W_c_ns = NaN(N,1);
W_h_ns = NaN(N,1);

for n = 1:N
    
    Wm_c_ns = NaN(M,1);
    Wt_c_ns = NaN(T,1);
    
    Wm_h_ns = NaN(M,1);
    Wt_h_ns = NaN(T,1);
    
    for t = 1:T
        for m = 1:M
            
            Wm_c_ns(m,1) = beta.^(m-1) * log(c_ns(m,t,n));
            Wm_h_ns(m,1) = beta.^(m-1) * kappa*h_ns(m,t,n).^(1+1/nu)./(1+1/nu);
            
        end
        
        Wt_c_ns(t,1) = beta.^(12*(t)) * sum(Wm_c_ns,1);
        Wt_h_ns(t,1) = beta.^(12*(t)) * sum(Wm_h_ns,1);
        
    end
    
    W_c_ns(n,1) = sum(Wt_c_ns,1);
    W_h_ns(n,1) = sum(Wt_h_ns,1);
    
end

W_ns=W_c_ns-W_h_ns;
%% 3) CEV - Decomposition of Welfare Effects

% log(1+g) = log(c')+ kappa*h'^2/2 - log(c) - kappa*h^2/2 
%          = log(c'/c) + kappa/2*(h'^2-h^2)
% 1+g = c'/c * exp(kappa/2*(h'^2-h^2))

% Thus:
% weff_c: log(1+gc) = log(c')+ kappa*h^2/2 - log(c) - kappa*h^2/2 
%                   = log(c'/c)
%           weff_c = c'/c-1
% weff_h: log(1+gh) = log(c')+ kappa*h'^2/2 - log(c') - kappa*h'^2/2 
%                   = kappa/2*( h'^2-h^2 )
%           weff_h = exp( kappa/2*(h'^2-h^2)  ) -1
% total weff: g=gc+gh

betagm=NaN(M,1);
betagt=NaN(T,1);

    for t = 1:40
        for m = 1:M
            
            betagm(m,1) = beta^(m-1);
            
        end
        
        betagt(t,1)=sum(betagm.^(12*t));
        
    end
betag = sum(betagt);


weff_c = NaN(N,1);
weff_h = NaN(N,1);

for n = 1:N
    
    weff_c(n,1) = exp(W_c_ns(n,1)-W_c(n,1)).^(1/betag) -1 ;
    weff_h(n,1) = exp(-W_h(n,1)+W_h_ns(n,1)).^(1/betag) -1 ; 
    
end

weff = weff_c+weff_h;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 4) Statistics

figure(1)
subplot(1,3,1)
histogram(weff)
title('Consumption Compensation Distribution')
figure(1)
subplot(1,3,2)
histogram(weff_c)
title('Consumption Effect')
subplot(1,3,3)
histogram(weff_h)
title('Labor Effect')


mean_w = mean(weff);
med_w = median(weff);
std_w = std(weff);
cv_w = (std_w/mean_w)*100;
max_w = max(weff);
min_w = min(weff);
p90_w = prctile(weff,90);

mean_c = mean(weff_c);
med_c = median(weff_c);
std_c = std(weff_c);
cv_c = (std_c/mean_c)*100;
max_c = max(weff_c);
min_c = min(weff_c);
p90_c = prctile(weff_c,90);

mean_h = mean(weff_h);
med_h = median(weff_h);
std_h = std(weff_h);
cv_h = (std_h/mean_h)*100;
max_h = max(weff_h);
min_h = min(weff_h);
p90_h = prctile(weff_h,90);

FID = fopen('wef_6.tex', 'w');
fprintf(FID, '\\begin{tabular}{|rrrrrrr|}\\hline \n');
fprintf(FID, 'mean & std & cv & med & max & min & p90 \\\\ \\hline \n');
for k=1:length(mean_w)
    fprintf(FID, '%8.2f & %8.2f & %8.2f & %8.2f  & %8.2f  & %8.2f  & %8.2f \\\\ ', mean_w(k), std_w(k), cv_w(k), med_w(k), max_w(k), min_w(k), p90_w(k));
    if k==length(mean_w)
        fprintf(FID, '\\hline ');
    end
    fprintf(FID, '\n');
end

fprintf(FID, '\\end{tabular}\n');
fclose(FID);

FID = fopen('wefc_6.tex', 'w');
fprintf(FID, '\\begin{tabular}{|rrrrrrr|}\\hline \n');
fprintf(FID, 'mean & std & cv & med & max & min & p90 \\\\ \\hline \n');
for k=1:length(mean_c)
    fprintf(FID, '%8.2f & %8.2f & %8.2f & %8.2f  & %8.2f  & %8.2f  & %8.2f \\\\ ', mean_c(k), std_c(k), cv_c(k), med_c(k), max_c(k), min_c(k), p90_c(k));
    if k==length(mean_c)
        fprintf(FID, '\\hline ');
    end
    fprintf(FID, '\n');
end

fprintf(FID, '\\end{tabular}\n');
fclose(FID);

FID = fopen('wefh_6.tex', 'w');
fprintf(FID, '\\begin{tabular}{|rrrrrrr|}\\hline \n');
fprintf(FID, 'mean & std & cv & med & max & min & p90 \\\\ \\hline \n');
for k=1:length(mean_h)
    fprintf(FID, '%8.2f & %8.2f & %8.2f & %8.2f  & %8.2f  & %8.2f  & %8.2f \\\\ ', mean_h(k), std_h(k), cv_h(k), med_h(k), max_h(k), min_h(k), p90_h(k));
    if k==length(mean_h)
        fprintf(FID, '\\hline ');
    end
    fprintf(FID, '\n');
end

fprintf(FID, '\\end{tabular}\n');
fclose(FID);

%Digits = 4;
%latex_table = latex(sym(ceq,'d'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

