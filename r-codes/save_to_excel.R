##setwd("C:/Users/PC/OneDrive/Ph.D/98. Paper/1. vix-forecast/forecasting-vix")
setwd("E:/OneDrive/Ph.D/98. Paper/1. vix-forecast/forecasting-vix")
##setwd("D:/OneDrive/Ph.D/98. Paper/1. vix-forecast/forecasting-vix")

load("NewResults.RData")


# 필요 패키지 설치 및 로드
install.packages("openxlsx")
library(openxlsx)

# .RData 파일 로드
load("NewResults.RData")

# 모든 객체 이름 가져오기
object_names <- c("rw1", "rw5", "rw10", "rw22"
                , "harx1", "harx5", "harx10", "harx22"
                , "har1", "har5", "har10", "har22"
                , "arx1", "arx5", "arx10", "arx22"
                , "rf1_14", "rf5_14", "rf10_14", "rf22_14")

# 빈 데이터프레임 생성
combined_data <- list()

# 각 객체에서 'pred' 항목을 추출하여 결합
for (name in object_names) {
  object <- get(name)  # 객체 가져오기
  
  # 객체에 'pred'라는 항목이 있는지 확인
  if ("pred" %in% names(object)) {
    pred_values <- object$pred  # 'pred' 열 추출
    
    # 데이터프레임에 추가
    combined_data <- cbind(combined_data, pred_values)
 #   colnames(combined_data)[ncol(combined_data)] <- name  # 열 이름을 객체 이름으로 설정
  }
}

combined_data = data.frame(combined_data)
colnames(combined_data) <-object_names

# 엑셀 파일로 저장
write.xlsx(combined_data, "NewResults.xlsx", rowNames = FALSE)

