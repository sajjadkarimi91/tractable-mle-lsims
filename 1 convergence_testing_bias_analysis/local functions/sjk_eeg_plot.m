function my_eeg_plot(EEG , scale_factor  , srate)


figure1 = figure('Position' ,  [200 200 850 450] );

MyTime = linspace(0,size(EEG,2)/srate , size(EEG,2));


grid on
hold off
n=1;
N = min(size(EEG , 1) , 256);

plot(MyTime , n+0*EEG(n,:)/scale_factor,'k','LineWidth',0.5)
hold on
plot(MyTime , n+EEG(n,:)/scale_factor,'b')
hold on

for n=2:N
    
    plot(MyTime , n+0*EEG(n,:)/scale_factor,'k','LineWidth',0.5)
    plot(MyTime , n+EEG(n,:)/scale_factor,'b')
    
    
end

xlim([MyTime(1),MyTime(end)])
ylim([1+min(EEG(n,:))/scale_factor - scale_factor/100 ,  n+ max(EEG(1,:))/scale_factor + scale_factor/100])


ax = gca;
ax.YAxisLocation = 'left';
ax.YTick = 1:N;


text_input = ['$',num2str(scale_factor),'~\mu V$'];

% Create textbox
annotation(figure1,'textbox',...
    [0.04 0.13  0.1023  0.0418],...
    'String',{text_input},...
    'LineStyle','none',...
    'Interpreter','latex',...
    'FitBoxToText','off','FontWeight','bold','FontSize',11);


annotation(figure1,'doublearrow',[0.107 0.107],...
    [0.16 0.13] , 'Head1Style' , 'rectangle', 'Head2Style' , 'rectangle' , 'Head1Width',2, 'Head2Width',2);


