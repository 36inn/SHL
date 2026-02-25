%%=  
function [transform_population] = transform(Population,offspring_Population,PopSize,T,T_ma,V,steps_of_job)
%%iuput
% Population 
% offspring_Population 
%% output 
%%
transform_population_p=zeros(PopSize,V+1); 
transform_population_o=zeros(PopSize,V+1); 
%transform_population=[]; 
  for index=1: PopSize
     [~,Cmax_p,~,~,~] = newDecode(T,T_ma,steps_of_job,Population{index,1});
    transform_population_p(index,:)=[Population{index,1}(1,:) Population{index,1}(2,:) Cmax_p];
     [~,Cmax_o,~,~,~] = newDecode(T,T_ma,steps_of_job,offspring_Population{index,1});
    transform_population_o(index,:)=[offspring_Population{index,1}(1,:) offspring_Population{index,1}(2,:)  Cmax_o];
  end
transform_population =[transform_population_p;transform_population_o];
transform_population=unique(transform_population,'rows'); 
%[~,b1,~]=unique(transform_population(:,V+1:V+2),'rows');  
%transform_population=transform_population(b1,:);

end