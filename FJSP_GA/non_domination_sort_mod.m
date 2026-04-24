
function z = non_domination_sort_mod(x, M, V)

[N, ~] = size(x);
z=zeros(N,M + V + 2);
front = 1;
F(front).f = [];      
Individual=cell(N,2); 
for i = 1 : N
    Individual{i,1}=0;
    for j = 1 : N
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        for k = 1 : M        
            if (x(i,V + k) < x(j,V + k))  
                dom_less = dom_less + 1;
            elseif (x(i,V + k) == x(j,V + k))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        if dom_less == 0 && dom_equal ~= M % 
            Individual{i,1} = Individual{i,1} + 1;
        elseif dom_more == 0 && dom_equal ~= M % 
            Individual{i,2} = [Individual{i,2} j];
        end
    end   
    if Individual{i,1} == 0 
        x(i,M + V + 1) = 1;
        F(front).f = [F(front).f i]; 
    end
end

while ~isempty(F(front).f)
    Q = []; 
    for i = 1 : length(F(front).f)
         temp= Individual{F(front).f(i),2};                    
        if ~isempty(temp)
            for j = 1 : length(temp) 
                Individual{temp(j),1}=Individual{temp(j),1} - 1; 
                     
     
                if Individual{temp(j),1}  == 0
                    x(temp(j),M + V + 1) = front + 1;
                    Q = [Q temp(j)];
                end
            end
        end
    end
   
   front =  front + 1;
   F(front).f = Q;
end
 
[~,index_of_fronts] = sort(x(:,M + V + 1));
sorted_based_on_front=x; 
for i = 1 : length(index_of_fronts)
    sorted_based_on_front(i,:) = x(index_of_fronts(i),:);
end
 
current_index = 0;
%% Crowding distance 
 
for front = 1 : (length(F) - 1)
    y = zeros(length(F(front).f),M+V+1); 
    previous_index = current_index + 1;
    
    for i = 1 : length(F(front).f)
        y(i,:) = sorted_based_on_front(current_index + i,:);
    end
    current_index = current_index + i;
    for i = 1 : M
        [~, index_of_objectives] = sort(y(:,V + i));
        sorted_based_on_objective = y;  
        for j = 1 : length(index_of_objectives)
            sorted_based_on_objective(j,:) = y(index_of_objectives(j),:);
        end
        f_max = sorted_based_on_objective(length(index_of_objectives), V + i);
        f_min = sorted_based_on_objective(1, V + i);                          
        y(index_of_objectives(length(index_of_objectives)),M + V + 1 + i)= Inf;
     
        y(index_of_objectives(1),M + V + 1 + i) = Inf;
        for j = 2 : length(index_of_objectives) - 1
            next_obj  = sorted_based_on_objective(j + 1,V + i);
            previous_obj  = sorted_based_on_objective(j - 1,V + i);
            if (f_max - f_min == 0) 
                y(index_of_objectives(j),M + V + 1 + i) = Inf;
            else
                y(index_of_objectives(j),M + V + 1 + i) = ...
                    (next_obj - previous_obj)/(f_max - f_min); 
            end
        end
    end

    distance = zeros(length(F(front).f),1);
    for i = 1 : M
        distance(:,1) = distance(:,1) + y(:,M + V + 1 + i);  
    end
    y(:,M + V + 2) = distance;
    
    y = y(:,1 : M + V + 2); 

     [~,index_of_y]=sort(y(:,M + V + 2),"descend"); 
    yy=y;
    for i=1:size(y,1)
        yy(i,:)=y(index_of_y(i),:);
    end


    z(previous_index:current_index,:) = yy;
    
end
