function f  = replace_chromosome(ndsm_Population, M, V,Popsize)

sorted_chromosome=zeros(size(ndsm_Population,1) ,M + V + 2);
[~,index] = sort(ndsm_Population(:,M + V + 1));
for i = 1 : size(ndsm_Population,1)    
    sorted_chromosome(i,:) = ndsm_Population(index(i),:);
end
 

max_rank = max(ndsm_Population(:,M + V + 1));
 

previous_index = 0;
for i = 1 : max_rank
    
    current_index = find(sorted_chromosome(:,M + V + 1) == i, 1, 'last' );

    if current_index > Popsize
       
        remaining = Popsize - previous_index;
        
        temp_pop = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
        
        [~,temp_sort_index] = ...
            sort(temp_pop(:, M + V + 2),'descend');
        
        for j = 1 : remaining
            f(previous_index + j,:) = temp_pop(temp_sort_index(j),:);
        end
       return;
    elseif current_index < Popsize
        
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
     else 
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
           return;
    end
    previous_index = current_index;
end
