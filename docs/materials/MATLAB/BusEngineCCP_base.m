%% Bus engine replacement: full-solution maximum likelihood
% Teaching illustration inspired by Rust (1987), not a replication.

clear; clc; close all

%% Static random utility model
Nstatic = 1000;
xStaticGrid = (0:0.9:18)';
thetaStaticTrue = [2; -0.15]; % continuation intercept and mileage coefficient
[xStatic,decisionStatic] = simulate_static_logit( ...
    Nstatic,thetaStaticTrue,xStaticGrid);

staticObjective = @(theta) static_logit_nll(theta,xStatic,decisionStatic);
staticOptions = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');
[thetaStaticHat,~,exitStatic] = fminunc(staticObjective,[0; 0],staticOptions);
assert(exitStatic > 0,'Static logit did not converge')

disp(table(thetaStaticTrue,thetaStaticHat, ...
    'RowNames',{'intercept','mileage'}))

%% Data and model
scriptFile = matlab.desktop.editor.getActiveFilename;
if isempty(scriptFile)
    scriptFile = mfilename('fullpath');
end
dataPath = fullfile(fileparts(scriptFile),'..','Data', ...
    'BusEngineCCP_data.csv');
data = readtable(dataPath); % simulated bus-period panel, not Rust's original data
% state: mileage-state index; type: fixed bus type; decision: 1 replace, 2 continue

x = (0:0.9:18)'; % mileage value associated with each state index
nx = length(x);
beta = 0.9; % fixed discount factor
thetaTrue = [2; -0.15; 1]; % simulation values: intercept, mileage, type 2

busExample = data(data.id==1,:);
replaceRows = busExample.decision==1;
figure
plot(busExample.period,busExample.mileage,'-o'); hold on
scatter(busExample.period(replaceRows),busExample.mileage(replaceRows),60,'filled')
xlabel('Period'); ylabel('Mileage')
title('Mileage and replacement decisions for bus 1')
legend('Mileage','Replace','Location','best')

F1 = zeros(nx); % rows are current states; columns are next states
F1(:,1) = 1;
F2 = (x' >= x).*exp(-(x'-x))*(1-exp(-0.9)); % continue: mileage weakly rises
F2(:,end) = 1-sum(F2(:,1:end-1),2); % terminal state collects remaining mass

assert(max(abs(sum(F1,2)-1)) < 1e-12)
assert(max(abs(sum(F2,2)-1)) < 1e-12)

disp('Top-left block of replacement transition matrix F1')
disp(F1(1:6,1:6))
disp('Top-left block of continuation transition matrix F2')
disp(F2(1:6,1:6))

%% Full-solution maximum likelihood
theta0 = [1; -0.1; 0.5];
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');
objective = @(theta) fiml(theta,data.state,data.type,data.decision, ...
    x,F1,F2,beta);
[thetaHat,~,exitflag] = fminunc(objective,theta0,options);
assert(exitflag > 0,'FIML did not converge')

disp(table(thetaTrue,thetaHat,'RowNames',{'intercept','mileage','type 2'}))

%% Functions
function nll = fiml(theta,state,type,decision,x,F1,F2,beta)
[v1,v2] = solve_model(theta,x,F1,F2,beta);
index = state + (type-1)*length(x); % map mileage and type to the value arrays
valueDiff = v2(index)-v1(index); % dynamic logit index: continue minus replace
softplus = max(valueDiff,0)+log1p(exp(-abs(valueDiff))); % log(1+exp(valueDiff))
nll = -sum((decision==2).*valueDiff-softplus);
end

function [v1,v2] = solve_model(theta,x,F1,F2,beta)
gamma = 0.5772156649;
V = zeros(length(x),2);
flowContinue = [theta(1)+theta(2)*x, ... % one column for each bus type
    theta(1)+theta(2)*x+theta(3)];

for iter = 1:1000
    v1 = beta*F1*V; % replace value: zero flow utility and reset mileage
    v2 = flowContinue+beta*F2*V; % continue value: flow plus continuation value
    m = max(v1,v2);
    Vnew = m+log(exp(v1-m)+exp(v2-m))+gamma;
    if max(abs(Vnew-V),[],'all') < 1e-8
        return
    end
    V = Vnew;
end
error('Value iteration did not converge')
end

function [x,decision] = simulate_static_logit(N,theta,xGrid)
rng(2026)
state = randi(length(xGrid),N,1);
x = xGrid(state); % observed mileage entering static continuation utility
index = theta(1)+theta(2)*x; % u_continue(x)-u_replace(x)
pContinue = 1./(1+exp(-index));
decision = 1+(rand(N,1)<pContinue); % 1 replace, 2 continue
end

function nll = static_logit_nll(theta,x,decision)
index = theta(1)+theta(2)*x; % no future-value term in the static model
softplus = max(index,0)+log1p(exp(-abs(index)));
nll = -sum((decision==2).*index-softplus);
end
