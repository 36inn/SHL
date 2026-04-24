function [T]=txt2matrix(filename)
%
%%
%filename='3128_10j_5m.txt';


A=dlmread(filename,' ');
job_number=A(1,1); 
machine_number=A(1,2);
if machine_number==0
    machine_number=A(1,3);
end

B=A;B(1,:)=[];
B(:, all(B==0)) = []; 
step_of_jobs=B(:,1); 
T=cell(job_number,1);
for i=1:job_number
    process_time=zeros(step_of_jobs(i),machine_number); 
    temp=B(i,:);temp(temp==0)=[];
    flag=2;  
    for index=1:step_of_jobs(i) 
        x=temp(flag);
        temptemp=temp(flag+1:flag+2*x);
        for j=1:x
            machine=temptemp(2*j-1);
            time=temptemp(2*j);
            process_time(index,machine)=time;
        end
        flag=flag+2*x+1;
    end
    T{i}=process_time;
end

T_x=T;
for i=1:length(T)
    T_x{i}=num2cell(T{i},2);  
end
T=T_x;

%filename = 'job_%d machine_%d';
%str = sprintf(filename,job_number,machine_number);
%save(str,"T_x");

%str = sprintf(filename(1:length(filename)-4));  
%save(str,"T");
end


