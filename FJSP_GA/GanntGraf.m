function GanntGraf(job_array,chromosome,MachineList,ST,PT,fit)
%% 
num_j=max(job_array); 
num_m=max(unique(MachineList)); 
num_op=numel(chromosome);
fit_n=floor(log10(fit))+1;
fit_str=num2str(fit);
if  numel(fit_str)>1
    if str2double(fit_str(2))>=5
        maxX=(10^(fit_n-1))*(str2double(fit_str(1))+1);
    elseif str2double(fit_str(2))<5
        maxX=(10^(fit_n-1))*(str2double(fit_str(1)))+(10^(fit_n-2)*5);
    end
elseif numel(fit_str)==1
    maxX=(10^(fit_n-1))*(str2double(fit_str(1))+1);
end
% xlim([0,30]);ylim([0 3.5]);
figure(2);
axis([0,maxX-0,0,num_m+0.5]);
set(gca,'xtick',0:maxX/10:maxX) ;
set(gca,'ytick',0:1:num_m+0.5) ;
xlabel('Time','FontName','Times New Roman','FontSize',15);
ylabel('Machine','FontName','Times New Roman','FontSize',15); 


%color=rand(num_j,3);
color=0.7*rand(num_j,3)+0.3;  
n_color=zeros(num_op,3);
for i=1:num_j
    counter_j(i)=1;
end
for i=1:length(chromosome)
    n_color(i,:)=color(chromosome(i),:);
end
% for i=1:num_j
%    pos=find(XX==job_array(i));
%    for j=1:length(pos)
%        n_color(pos(j),:)=color(i,:);
%    end
% end
rec=[0,0,0,0];%temp data space for every rectangle
for i = 1:num_op
    job=chromosome(i);
    pos_j=find(job_array==job);
    rec(1) = ST(i);
    rec(2) = MachineList(i)-0.2;  
    rec(3) = PT(i);  
    rec(4) = 0.5;
    txt=sprintf('%d-%d',job,counter_j(pos_j(1)));
    rectangle('Position',rec,'LineWidth',0.5,'LineStyle','-','faceColor',n_color(i,:));%draw every rectangle
    text(ST(i)+0.1,(MachineList(i)),txt,'FontWeight','Bold','FontSize',8,'Color','black');%label the id of every task 
    counter_j(pos_j(1))=counter_j(pos_j(1))+1;
end
%title('TITLE','FontName','Times New Roman','FontSize',15); 
hold on;grid on;box on;
plot([fit,fit],[0,num_m+0.5],'r','linewidth',2);
text(fit,0.5,['',num2str(fit)],'FontWeight','Bold','FontName','Times New Roman','FontSize',15);
end
