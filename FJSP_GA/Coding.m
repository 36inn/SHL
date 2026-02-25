function [Population,machine_of_job] = Coding(T,PopSize,steps_of_job)
%% variable declaration
num_of_jobs = length(T);                                                   %number of jobs
%num_of_machines = length(T{1}{1});                                         %number of machines

machine_of_job=cell(num_of_jobs,1); 
len_of_chromosome = sum(steps_of_job);    
% calculate the machine set for each steps of each job
for i=1:num_of_jobs
    steps=cell(steps_of_job(i),1);
    for j = 1:steps_of_job(i)
        machineset=[];
        for k=1:length(T{i}{j})
            if T{i}{j}(k)~=0
                machineset=[machineset k];
            end
        end
        steps{j}=machineset;
    end
    machine_of_job{i}=steps;
end
%% steps chromosome 
%Coding is based on the step of the job
step_chromsome=[];
for i = 1:num_of_jobs
    for j = 1:steps_of_job(i)
        step_chromsome=[step_chromsome i];
    end
end
step_population =zeros(PopSize,len_of_chromosome);
%Generate population with chromosome containing random genes
for i = 1:PopSize
    step_population(i,:)=step_chromsome(randperm(length(step_chromsome(:))));
end
%% 
machine_population =zeros(PopSize,len_of_chromosome);
for index = 1:PopSize  
    machine_chromosome=[];
    for i=1:num_of_jobs
        for j=1:steps_of_job(i)
            %{
                pos_n= randperm(length(machine_of_job{i}{j}),2);
                machine1=machine_of_job{i}{j}(pos_n(1));
                machine2=machine_of_job{i}{j}(pos_n(2));
                if (rand(1)<0.8) && (T{i}{j}(machine1)>T{i}{j}(machine2))
                    machine = machine2;
                else
                    machine = machine1;
                end
                machine_chromosome=[machine_chromosome machine];
            %}
            pos_n=randperm(length(machine_of_job{i}{j}),1);
            machine=machine_of_job{i}{j}(pos_n(1));
            machine_chromosome=[machine_chromosome machine];
        end
    end
    machine_population(index,:) = machine_chromosome;
end
%% 
Population=cell(PopSize,1);
for i=1:PopSize
    Population{i} =[step_population(i,:);machine_population(i,:)];
end


