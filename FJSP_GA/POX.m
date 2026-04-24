
  function [NewPopulation] = POX(T,T_ma,Population,PopSize,Pc,Pm,steps_of_job,machine_of_job)
%% INPUT:
%T--input matrix:
%  For example: 
%  A partial flexible scheduling problem in which 8 jobs are processed 
%  on 8 machines, in which the number of available machining machines per
%  step of job is less than or equal to the total number of machines
% J1 ={[5 3 5 3 3 0 10 9];[10 0 5 8 3 9 9 6];[0 10 0 5 6 2 4 5]};
% J2 ={[5 7 3 9 8 0 9 0];[0 8 5 2 6 7 10 9];[0 10 0 5 6 4 1 7];[10 8 9 6 4 7 0 0]};
% J3 ={[10 0 0 7 6 5 2 4];[0 10 6 4 8 9 10 0];[1 4 5 6 0 10 0 7]};
% J4 ={[3 1 6 5 9 7 8 4];[12 11 7 8 10 5 6 9];[4 6 2 10 3 9 5 7]}; 
% J5 ={[3 6 7 8 9 0 10 0];[10 0 7 4 9 8 6 0];[0 9 8 7 4 2 7 0];[11 9 0 6 7 5 3 6]};
% J6 ={[6 7 1 4 6 9 0 10];[11 0 9 9 9 7 6 4];[10 5 9 10 11 0 10 0]};
% J7 ={[5 4 2 6 7 0 10 0];[0 9 0 9 11 9 10 5];[0 8 9 3 8 6 0 10]};
% J8 ={[2 8 5 9 0 4 0 10];[7 4 7 8 9 0 10 0];[9 9 0 8 5 6 7 1];[9 0 3 7 1 5 8 0]};
%T={J1;J2;J3;J4;J5;J6;J7;J8}; 8*1 cell
% Population 
%PopSize--Population size in genetic algorithms,2*PopSize+1
%Pc--probability of crossover
%Pm--probability of mutation
% M
% V
%% OUTPUT
% NewPopulation--
%% variable declaration
num_of_jobs = length(steps_of_job);                                                   %number of jobs
len_of_chromosome = sum(steps_of_job);
%%  

  %% Selection 
    Parent=cell(PopSize,1);
    dex=0;
    while dex <PopSize
        pos = randperm(PopSize,2);
        chromosome1 = Population{pos(1)};chromosome2 = Population{pos(2)};
        [~,Cmax1,~,~,~] = newDecode(T,T_ma,steps_of_job,chromosome1);
        [~,Cmax2,~,~,~] = newDecode(T,T_ma,steps_of_job,chromosome2);
        if  Cmax1<Cmax2
            chromosome = chromosome1;
        else
            chromosome = chromosome2;
        end
        dex=dex+1;
        Parent{dex} = chromosome;
    end
    %
    %Parent{PopSize}=BestChromosome_cmax;  

   %% Crossover: IPOXfor steps, MPX for machine 
   
   Children_group1=cell(PopSize,1);
   for number=1:PopSize/2         %  number=1:(PopSize-2)
       %Parent individuals are selected for crossover operation:
       %Parent1 is selected sequentially and Parent2 is selected randomly.
       index_parent =randperm(PopSize,2);%index_parent = randi([1,PopSize]);
       Parent1=Parent{index_parent(1)};
       Parent2=Parent{index_parent(2)};
       Children1 = Parent1; 
       Children2 = Parent2;

       if rand(1)<=Pc %Use the probability to determine if crossover is required
           %% Part1: IPX for step
           %Randomly divide the set of jobs {1,2,3...,n} into two non-empty sub-sets J1 and J2.
           Children1(1,:)=0;Children2(1,:)=0;
           num_J1 = randi([1,num_of_jobs]);
           if num_J1==num_of_jobs
               num_J1 = fix(num_of_jobs/2);
           end
           J = randperm(num_of_jobs);
           J1 =J(1:num_J1);
           J2 =J(num_J1+1:num_of_jobs);
           % Copy the jobs that Parent1 contains in J1 to Children1,
           % and Parent2 contains in J2 to Children2, and keep them in place.
           for index = 1:num_J1                                            % look for jobs that Parent1 are included in J1
               job = J1(index);
               po=find(Parent1(1,:)==job);
               for j=1:length(po)
                   Children1(1,po(j))=Parent1(1,po(j));
               end
           end
           for index = 1:num_of_jobs-num_J1                                % look for jobs that Parent2 are included in J2
               job = J2(index);
               po=find(Parent2(1,:)==job);
               for j = 1:length(po)
                   Children2(1,po(j))=Parent2(1,po(j));
               end
           end
           %Copy the jobs that Parent1 contains in J1 to Children2,
           %and Parent2 contains in J2 to Children1 in their order.
           p=find(Children2(1,:)==0);v=1;
           for index = 1:len_of_chromosome                                            % look for jobs that Parent1 are included in J1
               job = Parent1(1,index);
               if  ~isempty(find(J1==job, 1))%ismember(job,J1)
                   gene=p(v);Children2(1,gene)=job;v=v+1;
               end
           end
           %
           p1=find(Children1(1,:)==0);v=1;
           for index = 1: len_of_chromosome                                            % look for jobs that Parent1 are included in J1
               job = Parent2(1,index);
               if  ~isempty(find(J2==job, 1)) %ismember(job,J2)
                   gene=p1(v);Children1(1,gene)=job;v=v+1;
               end
           end
           %%IPOX cross operation completed
           %% Part 2 UX for machine
           %

           for  i=1:num_of_jobs
               for j=1:steps_of_job(i)
                   if rand(1)<=Pc
                       gene=sum(steps_of_job(1:(i-1)))+j;
                       Children1(2,gene) = Parent2(2,gene);
                       Children2(2,gene) = Parent1(2,gene);
                   end
               end
           end
       end
      
       Children_group1{number}=Children1;Children_group1{number+PopSize/2}=Children2;

   end
%    Children_group1{PopSize-1}= BestChromosome_cmax;
%    Children_group1{PopSize}= BestChromosome_cost;
    %% Mutation 
    
      Children_group2=cell(PopSize,1);
    for iii=1:PopSize     %iii=1:(PopSize-2)
       aa=randi([1,PopSize]);
       temp_chromsome = Children_group1{aa};
       %% Mutation for steps
       if rand(1)<Pm
           for j=1:1
               pos1=randi([1,len_of_chromosome]); % Choose the sequence number of a gene to be mutated
               pos2=randi([1,len_of_chromosome]); %  Choose another the sequence number of a gene to be mutated
               Gene=temp_chromsome(1,pos1);
               temp_chromsome(1,pos1)=temp_chromsome(1,pos2);
               temp_chromsome(1,pos2)=Gene;
           end
           %% Mutation for machine          
           for k=1:randi(4)
               job = randi([1,num_of_jobs]) ;   %random choose job
               tempstep = randi([1,steps_of_job(job)]);   %random choose step
               
               %[~,pos_m]=min(T{job}{tempstep}(machine_of_job{job}{tempstep}));
               %newmachine = machine_of_job{job}{tempstep}(pos_m);
               
               newmachine = machine_of_job{job}{tempstep}(randi([1,length(machine_of_job{job}{tempstep})]));%random choose machine
               temp_chromsome(2,sum(steps_of_job(1:job-1))+tempstep)= newmachine;  %replace the old one
           end
       end
       %%
       Children_group2{iii} = temp_chromsome;    
    end
%    Children_group2{PopSize-1}= BestChromosome_cmax;
%    Children_group2{PopSize}= BestChromosome_cost;
   %% rebuild population
    NewPopulation=Children_group2;
 end

