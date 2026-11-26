%% Continuous Choices: Hansen-Singleton GMM teaching illustration
% This is not a replication of Hansen and Singleton (1982). It uses modern
% public data and simplified Treasury returns to illustrate Euler-equation GMM.

clear; clc; close all

%% Data
readLocal = true;
scriptFile = matlab.desktop.editor.getActiveFilename;
if isempty(scriptFile)
    scriptFile = mfilename('fullpath');
end
dataPath = fullfile(fileparts(scriptFile),'..','Data');
series = {'A794RX0Q048SBEA','DTB3','DGS10','CPIAUCSL'};
% A794RX0Q048SBEA: real consumption per capita, used to construct g_{t+1}.
% DTB3: 3-month Treasury bill discount rate, used for the short safe return.
% DGS10: 10-year Treasury yield, used for the simplified bond return.
% CPIAUCSL: consumer price index, used to convert returns from nominal to real.

if readLocal
    C = read_series(fullfile(dataPath,[series{1} '.csv']));
    bill = read_series(fullfile(dataPath,[series{2} '.csv']));
    bond = read_series(fullfile(dataPath,[series{3} '.csv']));
    cpi = read_series(fullfile(dataPath,[series{4} '.csv']));
else
    C = fred_series(series{1});
    bill = fred_series(series{2});
    bond = fred_series(series{3});
    cpi = fred_series(series{4});
end

C = retime(table2timetable(rmmissing(C)),'quarterly','lastvalue');
bill = retime(table2timetable(rmmissing(bill)),'quarterly','lastvalue');
bond = retime(table2timetable(rmmissing(bond)),'quarterly','lastvalue');
cpi = retime(table2timetable(rmmissing(cpi)),'quarterly','lastvalue');

C.Properties.VariableNames = {'consumption'};
bill.Properties.VariableNames = {'billYield'};
bond.Properties.VariableNames = {'bondYield'};
cpi.Properties.VariableNames = {'cpi'};
Data = synchronize(C,bill,bond,cpi,'intersection');
Data = Data(timerange(datetime(1962,1,1),datetime(2019,12,31),'closed'),:);

gAll = Data.consumption(2:end)./Data.consumption(1:end-1); % g_{t+1}=C_{t+1}/C_t
inflation = Data.cpi(2:end)./Data.cpi(1:end-1);            % gross CPI inflation

disc = Data.billYield(1:end-1)/100;                 % annualized bill discount yield
RfAll = (1./(1-disc*91/360))./inflation;            % real gross bill return R_{t+1,1}

y0 = Data.bondYield(1:end-1)/400;                   % quarterly yield at t
y1 = Data.bondYield(2:end)/400;                     % quarterly yield at t+1
RbAll = ((1+y1).^(-39)./(1+y0).^(-40))./inflation; % real gross bond return R_{t+1,2}

n = length(gAll)-1;
gC = gAll(2:end); % consumption growth entering the stochastic discount factor
R = [RfAll(2:end), RbAll(2:end)]; % two returns in the Euler pricing errors
X = [ones(n,1), RfAll(1:end-1), RbAll(1:end-1), gAll(1:end-1)]; % instruments Z_t

figure
tiledlayout(1,2)
nexttile; plot(Data.Time(2:end),gAll); yline(1); title('Consumption growth')
nexttile; plot(Data.Time(3:end),R); yline(1); title('Real gross returns')
legend('3-month bill','10-year bond','Location','best')

%% Two-step GMM
theta0 = [0.99; 5];
L = size(X,2);
W1 = eye(2*L);
options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton');

obj1 = @(theta) gmm_objective(theta,W1,gC,R,X);
[theta1,~,exit1] = fminunc(obj1,theta0,options);

U = stacked_moments(theta1,gC,R,X);
S = nw_cov(U,4);
W2 = S\eye(size(S));
obj2 = @(theta) gmm_objective(theta,W2,gC,R,X);
[theta,~,exit2] = fminunc(obj2,theta1,options);

g = mean(stacked_moments(theta,gC,R,X),1)';
D = numerical_jacobian(@(x) mean(stacked_moments(x,gC,R,X),1)',theta);
V = (D'*W2*D)\eye(2)/n;
se = sqrt(diag(V));
J = n*g'*W2*g;
df = 2*L-length(theta);
pValue = 1-chi2cdf(J,df);

fprintf('\nHansen-Singleton teaching illustration (quarterly)\n')
fprintf('beta  = %.4f  (se %.4f)\n',theta(1),se(1))
fprintf('gamma = %.4f  (se %.4f)\n',theta(2),se(2))
fprintf('J     = %.2f  (df %d, p %.3f)\n',J,df,pValue)
if exit1 <= 0 || exit2 <= 0
    warning('At least one GMM optimization did not report convergence.')
end
if theta(1) <= 0 || theta(1) >= 1 || theta(2) <= 0
    warning('The unrestricted estimate is outside beta in (0,1), gamma > 0.')
end

%% Functions
function T = read_series(file)
T = readtable(file);
T = T(:,1:2);
T.Properties.VariableNames = {'Time','Value'};
T.Time = datetime(T.Time,'InputFormat','yyyy-MM-dd');
end

function T = fred_series(id)
url = ['https://fred.stlouisfed.org/graph/fredgraph.csv?id=' id];
T = read_series(url);
end

function errors = pricing_errors(theta,gC,R)
sdf = theta(1)*gC.^(-theta(2));
errors = sdf.*R-1;
end

function U = stacked_moments(theta,gC,R,X)
errors = pricing_errors(theta,gC,R);
U = [X.*errors(:,1), X.*errors(:,2)];
end

function q = gmm_objective(theta,W,gC,R,X)
g = mean(stacked_moments(theta,gC,R,X),1)';
q = g'*W*g;
end

function S = nw_cov(U,L)
U = U-mean(U,1);
[n,~] = size(U);
S = U'*U/n;
for ell = 1:L
    G = U(ell+1:end,:)'*U(1:end-ell,:)/n;
    S = S+(1-ell/(L+1))*(G+G');
end
end

function D = numerical_jacobian(fun,theta)
g0 = fun(theta);
D = zeros(length(g0),length(theta));
for j = 1:length(theta)
    h = 1e-6*max(1,abs(theta(j)));
    step = zeros(size(theta)); step(j) = h;
    D(:,j) = (fun(theta+step)-fun(theta-step))/(2*h);
end
end
