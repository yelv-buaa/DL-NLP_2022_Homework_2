%% 深度学习与自然语言处理2022第二次大作业
%   吕晔
%   ZY2103518
%   EM算法的仿真验证
%   2022年4月15日

%% 自定义各项训练参数
%   define  pi1 = 0.4   第一枚硬币出现的概率
%   define  pi2 = 0.4   第二枚硬币出现的概率
%   define  pi3 = 0.2   第三枚硬币出现的概率
%   define  p = 0.3     第一枚硬币出现正面的概率
%   define  q = 0.3     第二枚硬币出现正面的概率
%   define  r = 0.8     第三枚硬币出现正面的概率
%   训练集样本数 N = 1000
%   EM循环次数  J = 5

%% 初始化数据
clc
clear

pi1=0.4;
pi2=0.4;
pi3=0.2;
p=0.3;
q=0.3;
r=0.8;
N=1000;
M=100;
J=5;
train_Data = zeros(N,M);
u1 = zeros(N,1);
u2 = zeros(N,1);
u3 = zeros(N,1);
theta = zeros(J,5);
theta(1,1)=0.2;
theta(1,2)=0.2;
theta(1,3)=0.5;
theta(1,4)=0.5;
theta(1,5)=0.9;

%% Prepare training data
Pi = [pi1,pi2,pi3];
pqr=[p,q,r];

for k=1:N
    r1 = rand;
    if(r1<Pi(1))
        coin_Flag = 1;
    elseif(r1<sum(Pi(1:2)))
        coin_Flag = 2;
    else
        coin_Flag = 3;  
    end

    for kk=1:M
        r2 = rand;
        if r2<pqr(coin_Flag)
            train_Data(k,kk) = 1;
        else
            train_Data(k,kk) = 0;
        end 

    end

    
end




%% start EM algorithm
for i=1:J-1
    

   %% E-Step : calculating u1,u2 and u3
%    u1=(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data).*theta(i,1))./(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data).*theta(i,1)+theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data).*theta(i,2)+theta(i,5).^train_Data.*(1-theta(i,5)).^(1-train_Data).*(1-theta(i,1)-theta(i,2)));
%    u2=(theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data).*theta(i,2))./(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data).*theta(i,1)+theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data).*theta(i,2)+theta(i,5).^train_Data.*(1-theta(i,5)).^(1-train_Data).*(1-theta(i,1)-theta(i,2)));
%    u3=(theta(i,5).^train_Data.*(1-theta(i,5)).^(1-train_Data).*(1-theta(i,1)-theta(i,2)))./(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data).*theta(i,1)+theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data).*theta(i,2)+theta(i,5).^train_Data.*(1-theta(i,5)).^(1-train_Data).*(1-theta(i,1)-theta(i,2)));
%    
%    u1=(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data))./(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data)+theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data)+theta(i,5).^train_Data.*(1-theta(i,5)).^(1-train_Data));
%    u2=(theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data))./(theta(i,3).^train_Data.*(1-theta(i,3)).^(1-train_Data)+theta(i,4).^train_Data.*(1-theta(i,4)).^(1-train_Data)+theta(i,5).^train_Data.*(1-theta(i,5)).^(1-train_Data));
%    u3=1-u1-u2;
   
%    u1=(theta(i,3).^sum(train_Data,2).*(1-theta(i,3)).^(M-sum(train_Data,2)))./(theta(i,3).^sum(train_Data,2).*(1-theta(i,3)).^(M-sum(train_Data,2))+theta(i,4).^sum(train_Data,2).*(1-theta(i,4)).^(M-sum(train_Data,2))+theta(i,5).^sum(train_Data,2).*(1-theta(i,5)).^(M-sum(train_Data,2)));
%    u2=(theta(i,4).^sum(train_Data,2).*(1-theta(i,4)).^(M-sum(train_Data,2)))./(theta(i,3).^sum(train_Data,2).*(1-theta(i,3)).^(M-sum(train_Data,2))+theta(i,4).^sum(train_Data,2).*(1-theta(i,4)).^(M-sum(train_Data,2))+theta(i,5).^sum(train_Data,2).*(1-theta(i,5)).^(M-sum(train_Data,2)));
%    u3=1-u1-u2;
%    
   u1=(theta(i,3).^sum(train_Data,2).*(1-theta(i,3)).^(M-sum(train_Data,2)).*theta(i,1))./(theta(i,3).^sum(train_Data,2).*(1-theta(i,3)).^(M-sum(train_Data,2)).*theta(i,1)+theta(i,4).^sum(train_Data,2).*(1-theta(i,4)).^(M-sum(train_Data,2)).*theta(i,2)+theta(i,5).^sum(train_Data,2).*(1-theta(i,5)).^(M-sum(train_Data,2)).*(1-theta(i,1)-theta(i,2)));
   u2=(theta(i,4).^sum(train_Data,2).*(1-theta(i,4)).^(M-sum(train_Data,2)).*theta(i,2))./(theta(i,3).^sum(train_Data,2).*(1-theta(i,3)).^(M-sum(train_Data,2)).*theta(i,1)+theta(i,4).^sum(train_Data,2).*(1-theta(i,4)).^(M-sum(train_Data,2)).*theta(i,2)+theta(i,5).^sum(train_Data,2).*(1-theta(i,5)).^(M-sum(train_Data,2)).*(1-theta(i,1)-theta(i,2)));
   u3=1-u1-u2;   
%    
   
%    for j=1:N
%        u1(j)=(theta(i,3).^train_Data(j).*(1-theta(i,3)).^(1-train_Data(j)).*theta(i,1))./(theta(i,3).^train_Data(j).*(1-theta(i,3)).^(1-train_Data(j)).*theta(i,1)+theta(i,4).^train_Data(j).*(1-theta(i,4)).^(1-train_Data(j)).*theta(i,2)+theta(i,5).^train_Data(j).*(1-theta(i,5)).^(1-train_Data(j)).*(1-theta(i,1)-theta(i,2)));
%        u2(j)=(theta(i,4).^train_Data(j).*(1-theta(i,4)).^(1-train_Data(j)).*theta(i,2))./(theta(i,3).^train_Data(j).*(1-theta(i,3)).^(1-train_Data(j)).*theta(i,1)+theta(i,4).^train_Data(j).*(1-theta(i,4)).^(1-train_Data(j)).*theta(i,2)+theta(i,5).^train_Data(j).*(1-theta(i,5)).^(1-train_Data(j)).*(1-theta(i,1)-theta(i,2)));
%        u3(j)=(theta(i,5).^train_Data(j).*(1-theta(i,5)).^(1-train_Data(j)).*(1-theta(i,1)-theta(i,2)))./(theta(i,3).^train_Data(j).*(1-theta(i,3)).^(1-train_Data(j)).*theta(i,1)+theta(i,4).^train_Data(j).*(1-theta(i,4)).^(1-train_Data(j)).*theta(i,2)+theta(i,5).^train_Data(j).*(1-theta(i,5)).^(1-train_Data(j)).*(1-theta(i,1)-theta(i,2)));
%     
%    end
%    

   %% M-Step : updata theta
   theta(i+1,1) = sum(u1)/N;
   theta(i+1,2) = sum(u2)/N;
   theta(i+1,3) = sum(u1.*sum(train_Data,2))/sum(u1.*M);
   theta(i+1,4) = sum(u2.*sum(train_Data,2))/sum(u2.*M);
   theta(i+1,5) = sum(u3.*sum(train_Data,2))/sum(u3.*M);
    
    
    
    
end














