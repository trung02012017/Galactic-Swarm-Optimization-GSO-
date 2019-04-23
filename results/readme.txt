
- Với mỗi hàm fitness f1, f2, ... fn lưu kết quả ra từng folder như trên.

- Thực nghiệm 1 là về accuracy và speed (time of convergence)
    Thì lưu tất cả kết quả của các models vào cùng 1 file: models_log.csv 
    Mỗi model với từng bộ tham số cũng phải lưu lại cái gbest sau từng vòng lặp của nó : vd như file: error_GA_[3,4,4].csv
    
- TN2 là về stability thì lúc này mỗi 1 model (với từng bộ tham số) phải chạy n lần (chọn luôn n = 20) đi.
    Với TN này thì lại không cần lưu cái error của từng model như bên trên (lưu làm gì vì không dùng đến)
    Riêng đối với TN này thì từng model sẽ có từng file csv riêng. vd như: stability_ga.csv , stability_pso.csv 
    

