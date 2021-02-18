close all
clear
clc

load FlatBeam_ARMAX.mat

%%

u = utrain;
y = ytrain;

N = length(u);

nk = 3;

PHI = u(nk:N);

for ii=1:nk-1
    PHI = [PHI, u(nk-ii:N-ii)];
end

for ii=1:nk-1
    PHI = [PHI, -y(nk-ii:N-ii)];
end

%PHI=[u(3:N), u(2:N-1), u(1:N-2), -y(2:N-1), -y(1:N-2)];
Y = y(nk:N);

theta = PHI\Y;
Ghat=tf(theta(1:3)',[1 theta(4:5)'],1);

disp('Comparing Bode plots of models with real system')
figure(8)
w=linspace(0,pi,N/2+1);
[mghat,pghat]=bode(Ghat,w);

subplot(2,1,1)
l=plot(w,mghat(:),'g-');figure(gcf);
set(l,'linewidth',1.5);
title(['Amplitude Bode plot'])
ylabel('mag  [gain]')
xlabel('w  [rad/s]');
legend('G_h_a_t','G_0')
axis([0 3.5 0 6])
grid on

subplot(2,1,2)
l=plot(w,pghat(:),'g-');figure(gcf);
set(l,'linewidth',1.5);
title(['Phase'])
ylabel('phase  [degree]')
xlabel('w  [rad/s]');
legend('G_h_a_t','G_0')
axis([0 3.5 -100 100])
grid on

yhat_armax = lsim(Ghat,ufull);

figure
plot(yhat_armax)
hold on
plot(yfull)

save yhat_full_armax_3nk.mat yhat_armax

%saveas(gcf,'LS_model.png')