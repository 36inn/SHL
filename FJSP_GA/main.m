%% NSGA-II for FJSP
clear; clc

%load('Mk01.mat')
filename='Mk01.txt';
[T]=txt2matrix(filename);
%%
M=1;      
Pc=0.7;  
Pm=0.4;  
PopSize=100; 
gen=100;  

%% 
num_of_jobs = length(T);                         
number_of_mas=length(T{1}{1});                  
steps_of_job =zeros(num_of_jobs,1);               
for i = 1:num_of_jobs
    steps_of_job(i,1)=length(T{i});
end
len_of_chromosome = sum(steps_of_job);            
V=2*len_of_chromosome;  
T_ma=[];
for i=1:length(T)
    T_ma=[T_ma;cell2mat(T{i})];  
end
%%
tic
[Population,machine_of_job] = Coding(T,PopSize,steps_of_job);  
%%
result_cmax=zeros(1,gen);

%% 
for i = 1 : gen
    %% 
    [offspring_Population] = POX(T,T_ma,Population,PopSize,Pc,Pm,steps_of_job,machine_of_job);

    %% 
    [transform_Population] = transform(Population,offspring_Population,PopSize,T,T_ma,V,steps_of_job);
    %% 
    %
    ndsm_Population=non_domination_sort_mod(transform_Population, M, V);
    % 
    new_population_tr = replace_chromosome(ndsm_Population, M, V,PopSize);
    %% 
    
    for index=1:PopSize
        Population{index} =[new_population_tr(index,1:V/2);new_population_tr(index,1*V/2+1:V)];
    end
    %% 

    [~,temp]=min(new_population_tr(:,V+1));result_cmax(1,i)=new_population_tr(temp,V+1);clear temp;
    %
    disp(i);
end

%% 
set(0,'defaultfigurecolor','w');

plot(result_cmax);xlabel('iter'); ylabel('makespan');
[Jobs,Cmax,MachineList,ST,PT] = newDecode(T,T_ma,steps_of_job,Population{1,1});
%GanntGraf(Jobs,Population{1,1}(1,:),MachineList,ST,PT,Cmax) ; 

toc
t=toc;  