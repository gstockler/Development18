%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Development - PS2                                    %
%                                                       %
%  Gabriela Barbosa                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; 
clc; 

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       QUESTION 2 - Welfare Costs of Seasons           %
%                     Stochastic Case                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   We want to analyze all possible combinations of deterministic
% and stochastic seasonal components, and also the nonseasonal
% consumption risk.
%   This means that we have to analzye 9 cases:
%       low+low; low+md; low+high; 
%       mid+low; mid+md; mid+high; 
%       high+low; high+md; high+high; 
%   In order to make this code user-friendly, I do not specify all
% possible combinations here. One need just to modify the choice
% on the risk matrix.
%   In the code it is pointed out where to modify for each case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%          Welfare gains of no nonseasonal risk

% 0) Set up

N = 1000; % number of individuals
T = 40; % number of periods (years) individual lives
M=12; % number of months in each period
ages=linspace(16,55,T); % ages throughout life

%%% Parameters

eta = 1; % coef. relative risk aversion
beta=.99; % discount factor

% idiosyncratic nonseasonal stochastic component; per period
sigma_eps=0.2;
log_eps = normrnd(0,sigma_eps,1,T);

% initial permanent consumption level variance; at the beggining of life
sigma_u=0.2;

% deterministic seasonal component of consumption; common among agents
g=NaN(12,3); % three levels: low, mid and high
g(:,2)=[.863; .691; 1.151; 1.140; 1.094; 1.060; 1.037; 1.037; 1.037; 1.002; .968; .921  ]; 
g(:,3)=[.727; .381; 1.303; 1.280; 1.188; 1.119; 1.073; 1.073; 1.073; 1.004; .935; .843 ];
g(:,1)=[.932; .845; 1.076; 1.070; 1.047; 1.030; 1.018; 1.018; 1.018; 1.001; .984; .961 ];

% stochastic seasonal component; common among agents
sigma_m=NaN(12,3); % three levels: low, mid and high
sigma_m(:,2)=[.085; .068; .29; .283; .273; .273; .239; .205; .188; .188; .171; .137 ]; 
sigma_m(:,3)=[.171; .137; .580; .567; .546; .546; .478; .410; .376; .376; .341; .273 ];
sigma_m(:,1)=[.043; .034; .145; .142; .137; .137; .119; .102; .094; .094; .085; .068 ];

% SEASON CHOICE
% deterministic
s=3; %1,2,3: low, med, high

% stochastic
ss=3; %1,2,3: low, med, high

%% 1) Welfare of det-season + stoc-season + risk

%%% Consumption loop
z=NaN(N,1);
c=NaN(M,T,N);

for n = 1:N
    
    log_u = normrnd(0,sigma_u,1); 
    z(n,1) = exp(-sigma_u./2)*exp(log_u);
    
    for t = 1:T
        for m = 1:M
            
            % CHOICE for stochastic season: sigma_m(:,L/M/H)
            log_eps_m = normrnd(0,sigma_m(m,ss),1); %m=1/low; 2/mid; 3/high
            
            % CHOICE for deterministic season: g(:,L/M/H)
            c(m,t,n) = z(n,1)*g(m,s)*exp(-sigma_m(m,ss)./2)*exp(log_eps_m)*exp(-sigma_eps./2)*exp(log_eps(t));
        
        end
    end
end

%%% Lifetime utility loop
W = NaN(N,1);

for n = 1:N
    
    Wm = NaN(M,1);
    Wt = NaN(T,1);
    
    for t = 1:40
        for m = 1:M
            
            if eta == 1
                
               Wm(m,1) = beta.^(m-1) .*  log( c(m,t,n) ); 
                
            else
                
                Wm(m,1) = beta.^(m-1) .*  ( c(m,t,n).^(1-eta) )./(1-eta);
            
            end
        end
        
        Wt(t,1) = beta^(12 * t) .* sum(Wm,1);
        
    end
    
    W(n,1) = sum(Wt,1);
    
end

%% 2) Welfare of det-season + stoc-season + NO risk

%%% Cnsumption
c_ns=NaN(M,T,N);

for n = 1:N
    for t = 1:T
        for m = 1:M
            
            % CHOICE for stochastic season: sigma_m(:,L/M/H)
            log_eps_m = normrnd(0,sigma_m(m,ss),1); %m=1/low; 2/mid; 3/high
            
            % CHOICE for deterministic season: g(:,L/M/H)
            c_ns(m,t,n) = z(n,1)*g(m,s)*exp(-sigma_m(m,ss)./2)*exp(log_eps_m)*exp(-sigma_eps./2)*exp(log_eps(t));

        end
    end
end


%%% Lifetime utility
W_ns = NaN(N,1);

for n = 1:N
    
    Wm_ns = NaN(M,1);
    Wt_ns = NaN(T,1);
    
    for t = 1:40
        for m = 1:M
            
            if eta == 1
                
               Wm_ns(m,1) = beta.^(m-1) .*  log( c_ns(m,t,n) ); 
                
            else
                
                Wm_ns(m,1) = beta.^(m-1) .*  ( c_ns(m,t,n).^(1-eta) )./(1-eta);
            
            end
        end
        
        Wt_ns(t,1) = beta.^(12.*t) .* sum(Wm_ns,1);
        
    end
    
    W_ns(n,1) = sum(Wt_ns,1);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3) Consumption equivalence

betagm=NaN(M,1);
betagt=NaN(T,1);

    for t = 1:40
        for m = 1:M
            
            betagm(m,1) = beta^(m-1);
            
        end
        
        betagt(t,1)=sum(betagm.^(12*t));
        
    end
betag = sum(betagt);

weff = NaN(N,1);

for n = 1:N
    
    if eta == 1
        weff(n,1) = exp( W_ns(n,1) - W(n,1) )^(1/betag)-1;
    else
        weff(n,1) = ( W_ns(n,1)./W(n,1) ).^(1/(1-eta))-1;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 4) Statistics
%figure(1)
%h1=histogram(W,'EdgeAlpha',0.1)
%hold on
%h2=histogram(W_ns,'EdgeAlpha',0.1)
%h1.NumBins=15
%h2.NumBins=15
%axis tight
%legend('Season+risk','Season+No risk','location','northwest')
%legend boxoff
%title('Welfare Distribution')

%figure(2)
h3=histfit(weff)
title('Consumption Compensation Distribution')
mean = mean(weff);
med = median(weff);
std = std(weff);
cv = (std/mean)*100;
max = max(weff);
min = min(weff);
p90 = prctile(weff,90);

FID = fopen('hh_nr.tex', 'w');
fprintf(FID, '\\begin{tabular}{|rrrrrrr|}\\hline \n');
fprintf(FID, 'mean & std & cv & med & max & min & p90 \\\\ \\hline \n');
for k=1:length(mean)
    fprintf(FID, '%8.2f & %8.2f & %8.2f & %8.2f  & %8.2f  & %8.2f  & %8.2f \\\\ ', mean(k), std(k), cv(k), med(k), max(k), min(k), p90(k));
    if k==length(mean)
        fprintf(FID, '\\hline ');
    end
    fprintf(FID, '\n');
end

fprintf(FID, '\\end{tabular}\n');
fclose(FID);

%Digits = 4;
%latex_table = latex(sym(ceq,'d'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

