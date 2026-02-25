function [Jobs,Cmax,MachineList,ST,PT] = newDecode(T,T_ma,steps_of_job,Chromosome)
%%
       % steps_of_job 
       % Chromosome 
%        
    %Jobs 
    %Cmax 
    %MachineList--The machine sequences corresponding to chromosome 
    %ST --the start time for each job step in chromosome 
    %PT --The operation time for each job step in chromome 
%%
num_of_jobs = length(T);                                                   %number of jobs
num_of_machines =length(T{1}{1});                                         %number of machines
len_of_chromosome = length(Chromosome);
StepList = zeros(1,len_of_chromosome); % 
step_chromsome=Chromosome(1,:); %


StepList_M=zeros(num_of_jobs,1); %
for i=1:len_of_chromosome
     job=step_chromsome(i);
     StepList_M(job)=StepList_M(job)+1;
     StepList(i)=StepList_M(job);
end

MachineList = zeros(1,len_of_chromosome);
ST = zeros(1,len_of_chromosome);
PT = zeros(1,len_of_chromosome);

      
%% Caculate MachineList and PT

   %
   s=0;test=zeros(1,num_of_jobs); %
   for i = 1:num_of_jobs
       test(i)=s+steps_of_job(i);
       s=test(i);
   end
   testtest=[0,test]; %[0,test(1:(num_of_jobs-1))]; %    
 %%
 for index = 1:len_of_chromosome
     postion = StepList(index); 
     x_1=Chromosome(1,index); 
     temp_x=testtest(x_1)+postion; 
     x_2=Chromosome(2,temp_x); 
     MachineList(index)=x_2;
     PT(index)=T_ma(temp_x,x_2);
 end

   
%% Caculate ST
%Machines = unique(MachineList);

Jobs=1:num_of_jobs;%Jobs = unique(Chromosome(1,:)); 

job_start_time = zeros(num_of_jobs,1);
job_end_time = zeros(num_of_jobs,1);
%machine_start_time = cell(num_of_machines,1);
machine_end_time   = zeros(num_of_machines,1);
machine_state = zeros(1,num_of_machines);                                %0--FirstWork;1--NotFirst 
%{
Machines_local=zeros(num_of_machines,1); %MachineList
for  i=1:length(Machines)
   Machines_local(Machines(i)) =i;
end
%}
for index = 1:len_of_chromosome
    job = Chromosome(1,index);
    machine = MachineList(index);
    pt=PT(index);
    step = StepList(index);

    if step==1                                                             %first step without considering the constrains between steps of same job
        if machine_state(machine)==0                                         % The machine is first used
            job_start_time(job)=0;
            job_end_time(job)=job_start_time(job)+pt;
        else
            job_start_time(job)=machine_end_time(machine);
            job_end_time(job)=job_start_time(job)+pt;
        end

    else
        if machine_state(machine)==0                                         % The machine is first used
            job_start_time(job)=job_end_time(job);
            job_end_time(job)=job_start_time(job)+pt;

        else
            job_start_time(job)=max(machine_end_time(machine),job_end_time(job));
            job_end_time(job)=job_start_time(job)+pt;
        end

    end
    machine_end_time(machine) =job_start_time(job)+pt;
    machine_state(machine)=1;
    ST(index)=job_start_time(job);
end

%% Caculate Cmax
Cmax = max(job_end_time(:,1));


