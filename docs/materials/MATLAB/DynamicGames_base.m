%% Dynamic entry and exit game: two-step CCP estimation
% Teaching illustration, not a replication of a particular application.

clear; clc; close all
rng(19)

%% Model and state space
Nm = 1000;
I = 5;                 % possible firms in each market
N = I-1;               % rivals faced by a focal firm
T = 10;
burnIn = 10;
S = 5;
Xn = 10;
beta = 0.9;

thetaTrue = [0; -0.05; 0.25; -1.5; -0.2];
% constant, permanent state, transitory state, entry cost, active rivals

stayProb = 0.7;
trans = stayProb*eye(S)+(1-stayProb)/(S-1)*(1-eye(S));
cdfTrans = cumsum(trans,2);

stateArgs = {0:N,0:Xn-1,0:S-1,0:1};
ii = 4:-1:1;
[stateGrid{ii}] = ndgrid(stateArgs{ii});
stateGrid = reshape(cat(5,stateGrid{:}),[],4);
% Columns: lagged active rivals, permanent state, transitory state, own lagged activity

choose = zeros(N+1);
for n = 0:N
    for k = 0:n
        choose(n+1,k+1) = nchoosek(n,k);
    end
end

utilityIndex = thetaTrue(1)+thetaTrue(2)*stateGrid(:,2) ...
    +thetaTrue(3)*stateGrid(:,3)+thetaTrue(4)*(1-stateGrid(:,4)) ...
    +thetaTrue(5)*stateGrid(:,1);

%% Simulate choices and states
pActive = solve_equilibrium(utilityIndex,trans,N,S,choose,beta,stateGrid);
[active,xState,sState,lagActive] = simulate_game( ...
    pActive,cdfTrans,S,Xn,I,Nm,T,burnIn,stateGrid);

sObs = repmat(sState,[1,1,I]);
sObs = sObs(:);
xObs = repmat(xState,[1,T,I]);
xObs = xObs(:);

lagTotal = sum(lagActive,3);
lagRivals = repmat(lagTotal,[1,1,I]);
lagRivals = lagRivals(:);
ownLag = lagActive(:);
lagRivals = lagRivals-ownLag;
action = active(:);

Z = [ones(I*Nm*T,1),xObs,sObs,1-ownLag,lagRivals];

%% First stage: estimate active-choice probabilities
W = [xObs,(xObs/Xn).^2,ownLag,lagRivals,(lagRivals/N).^2, ...
    sObs,sObs.*xObs/Xn,ownLag.*sObs,lagRivals.*sObs/(N*S)];
lambdaHat = mnrfit(W,categorical(action>0.5));

Wgrid = [stateGrid(:,2),(stateGrid(:,2)/Xn).^2,stateGrid(:,4), ...
    stateGrid(:,1),(stateGrid(:,1)/N).^2,stateGrid(:,3), ...
    stateGrid(:,3).*stateGrid(:,2)/Xn, ...
    stateGrid(:,4).*stateGrid(:,3), ...
    stateGrid(:,1).*stateGrid(:,3)/(N*S)];
ccpHat = mnrval(lambdaHat,Wgrid);
pActiveHat = ccpHat(:,2);

%% Rivals' choices and the continuation adjustment
gamma = double(eulergamma);
rivalProb = zeros(size(stateGrid,1),N+1);
fv = zeros(size(stateGrid,1),1);

for j = 1:size(stateGrid,1)
    nIncumbent = stateGrid(j,1);
    ownLagged = stateGrid(j,4);

    [~,entryState] = ismember([min(nIncumbent+ownLagged,N), ...
        stateGrid(j,2:3),0],stateGrid,'rows');
    [~,incumbentState] = ismember([max(nIncumbent+ownLagged-1,0), ...
        stateGrid(j,2:3),1],stateGrid,'rows');

    rivalProb(j,:) = rival_count_distribution(pActiveHat(entryState), ...
        pActiveHat(incumbentState),N-nIncumbent,nIncumbent,choose,N);
end

assert(max(abs(sum(rivalProb,2)-1)) < 1e-10)

for j = 1:size(stateGrid,1)
    expectedLogPInactive = 0;
    for sNext = 0:S-1
        nextStates = ismember(stateGrid,[(0:N)',repmat( ...
            [stateGrid(j,2),sNext,1],[N+1,1])],'rows');
        expectedLogPInactive = expectedLogPInactive ...
            +trans(stateGrid(j,3)+1,sNext+1)*rivalProb(j,:) ...
            *log(1-pActiveHat(nextStates));
    end
    fv(j) = beta*(gamma-expectedLogPInactive);
end

[matched,stateIndex] = ismember([lagRivals,xObs,sObs,ownLag],stateGrid,'rows');
assert(all(matched),'An observed state was not found in the state grid')

FV = fv(stateIndex);
Z(:,5) = rivalProb(stateIndex,:)*(0:N)';

%% Second stage: estimate payoff parameters
objective = @(theta) sum(log(1+exp(Z*theta+FV)) ...
    -action.*(Z*theta+FV));
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');
[thetaHat,~,exitflag] = fminunc(objective,0.1*ones(5,1),options);
assert(exitflag > 0,'Second-stage estimation did not converge')

disp(table(thetaTrue,thetaHat,'RowNames', ...
    {'constant','permanent state x','transitory state s', ...
    'entry cost','active rivals'}))

%% Functions
function p = solve_equilibrium(utilityIndex,trans,N,S,choose,beta,stateGrid)
gamma = double(eulergamma);
p = 1./(1+exp(-utilityIndex));

for iter = 1:10000
    pOld = p;
    for j = 1:size(stateGrid,1)
        nIncumbent = stateGrid(j,1);
        ownLagged = stateGrid(j,4);
        [~,entryState] = ismember([min(nIncumbent+ownLagged,N), ...
            stateGrid(j,2:3),0],stateGrid,'rows');
        [~,incumbentState] = ismember([max(nIncumbent+ownLagged-1,0), ...
            stateGrid(j,2:3),1],stateGrid,'rows');

        rivalProb = rival_count_distribution(p(entryState),p(incumbentState), ...
            N-nIncumbent,nIncumbent,choose,N);
        sameState = ismember(stateGrid,[(0:N)',repmat( ...
            stateGrid(j,2:4),[N+1,1])],'rows');
        expectedFlow = rivalProb*utilityIndex(sameState);

        expectedLogPInactive = 0;
        for sNext = 0:S-1
            nextStates = ismember(stateGrid,[(0:N)',repmat( ...
                [stateGrid(j,2),sNext,1],[N+1,1])],'rows');
            expectedLogPInactive = expectedLogPInactive ...
                +trans(stateGrid(j,3)+1,sNext+1)*rivalProb ...
                *log(1-p(nextStates));
        end
        index = expectedFlow+beta*(gamma-expectedLogPInactive);
        p(j) = 1/(1+exp(-index));
    end
    if max(abs(p-pOld)) < 1e-10
        return
    end
end
error('Equilibrium iteration did not converge')
end

function prob = rival_count_distribution(pEntry,pIncumbent, ...
    nEntry,nIncumbent,choose,N)
entryProb = choose(nEntry+1,1:nEntry+1) ...
    .*pEntry.^(0:nEntry).*(1-pEntry).^(nEntry-(0:nEntry));
incumbentProb = choose(nIncumbent+1,1:nIncumbent+1) ...
    .*pIncumbent.^(0:nIncumbent) ...
    .*(1-pIncumbent).^(nIncumbent-(0:nIncumbent));
prob = conv(entryProb,incumbentProb);
prob(end+1:N+1) = 0;
end

function [active,x,state,lagActive] = simulate_game( ...
    pActive,cdfTrans,S,Xn,I,Nm,T,burnIn,stateGrid)
active = zeros(Nm,T+burnIn,I);
lagActive = zeros(Nm,T+burnIn+1,I);
x = randi(Xn,Nm,1)-1;
state = zeros(Nm,T+burnIn+1);
state(:,1) = randi(S,Nm,1);
actionDraw = rand(Nm,T+burnIn,I);
stateDraw = rand(Nm,T+burnIn);

for m = 1:Nm
    nActive = 0;
    for t = 1:T+burnIn
        for i = 1:I
            [~,stateIndex] = ismember([nActive-lagActive(m,t,i), ...
                x(m),state(m,t)-1,lagActive(m,t,i)],stateGrid,'rows');
            active(m,t,i) = pActive(stateIndex)>actionDraw(m,t,i);
        end
        nActive = sum(active(m,t,:));
        lagActive(m,t+1,:) = active(m,t,:);
        state(m,t+1) = 1+sum(stateDraw(m,t)> ...
            cdfTrans(state(m,t),1:S-1));
    end
end

active = active(:,burnIn+1:T+burnIn,:);
state = state(:,burnIn+1:T+burnIn)-1;
lagActive = lagActive(:,burnIn+1:T+burnIn,:);
end
