clc;clear;close all;
pathMatrix = csvread("./data_csv/pathMatrix.csv");
obsMatrix = csvread("./data_csv/obs_trace.csv");
start = csvread("./data_csv/start.csv");
goal = csvread("./data_csv/goal.csv");
cylinderR = csvread("./data_csv/cylinder_r.csv");           % 动态障碍物的半径
cylinderH = csvread("./data_csv/cylinder_h.csv");
scatter3(start(1),start(2),start(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(goal(1),goal(2),goal(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')
text(start(1),start(2),start(3),'  Start','FontName','Times New Roman','FontWeight','bold');
text(goal(1),goal(2),goal(3),'  End','FontName','Times New Roman','FontWeight','bold');
xlabel('x(m)','FontWeight','bold'); ylabel('y(m)','FontWeight','bold'); zlabel('z(m)','FontWeight','bold');
title('UAV trajectory planning path','FontName','Times New Roman','FontWeight','bold'); axis equal;
set(gca,'fontsize',16,'FontName','Times New Roman','FontWeight','bold');%设置坐标轴字体大小
set(gca,'fontsize',16,'FontName','Times New Roman','FontWeight','bold');%设置坐标轴字体大小
timeStep = 0.1;
[n,~] = size(pathMatrix);
for i = 1:n-1
    obsCenter = [obsMatrix(i,1),obsMatrix(i,2),obsMatrix(i,3)];
    try delete(B1), catch, end
    try delete(B2), catch, end
    B1 = drawCylinder(obsCenter, cylinderR, cylinderH);
    B2 = scatter3(pathMatrix(i,1),pathMatrix(i,2),pathMatrix(i,3),80,'filled',"^",'MarkerFaceColor','g'...
                  ,'MarkerEdgeColor','k');
    if i >1
        b1 = plot3([obsMatrix(i-1,1),obsMatrix(i,1)],[obsMatrix(i-1,2),obsMatrix(i,2)]...
              ,[obsMatrix(i-1,3),obsMatrix(i,3)],'LineWidth',2,'color','b');
    end
    drawnow;
    b2 = plot3([pathMatrix(i,1),pathMatrix(i+1,1)],[pathMatrix(i,2),pathMatrix(i+1,2)],[pathMatrix(i,3),pathMatrix(i+1,3)],'LineWidth',2,'Color','r');
    if i == 2
        legend([b1,b2,B2],["Obstacle trajectory","UAV planning path","UAV"],'FontName','Times New Roman','FontWeight','bold','AutoUpdate','off','Location','best')
    end
end
%% 计算GS,LS,L
pathLength = 0;
for i=1:length(pathMatrix(:,1))-1, pathLength = pathLength + distanceCost(pathMatrix(i,1:3),pathMatrix(i+1,1:3)); end
fprintf("航路长度为:%f\n GS:%f °\n LS:%f °",pathLength, calGs(pathMatrix)/pi*180, calLs(pathMatrix)/pi*180);
%% 函数
% 球绘制函数
function bar = drawSphere(pos, r)
[x,y,z] = sphere(60);
bar = surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end
function bar = drawCylinder(pos, r, h)
[x,y,z] = cylinder(r,40);
z(2,:) = h;
bar = surfc(x + pos(1),y + pos(2),z,'FaceColor','interp');hold on;

% theta = linspace(0,2*pi,40);
% X = r * cos(theta) + pos(1);
% Y = r * sin(theta) + pos(2);
% Z = ones(size(X)) * h;
% fill3(X,Y,Z,[0 0.5 1]); % 顶盖
% fill3(X,Y,zeros(size(X)),[0 0.5 1]); % 底盖
end
% 欧式距离求解函数
function h=distanceCost(a,b)
h = sqrt(sum((a-b).^2, 2));
end