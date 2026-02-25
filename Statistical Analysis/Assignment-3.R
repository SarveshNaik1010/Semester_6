############################################################
# Weather Dataset - Descriptive Statistics & Data Cleaning
############################################################

# -----------------------------
# 1. Load Datasets
# -----------------------------

data1 <- read.csv("./Statistical Analysis/pune-1.csv")
data2 <- read.csv("./Statistical Analysis/pune-2.csv")
data3 <- read.csv("./Statistical Analysis/pune-3.csv")

# -----------------------------
# 2. Data Cleaning - Missing Values
# -----------------------------

# Check missing values
cat("Missing values in Dataset 1:\n")
print(colSums(is.na(data1)))

cat("Missing values in Dataset 2:\n")
print(colSums(is.na(data2)))

cat("Missing values in Dataset 3:\n")
print(colSums(is.na(data3)))

# Remove missing values
data1 <- na.omit(data1)
data2 <- na.omit(data2)
data3 <- na.omit(data3)

# -----------------------------
# 3. Mode Function
# -----------------------------

mode_function <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# -----------------------------
# 4. Function to Calculate All Measures
# -----------------------------

calculate_statistics <- function(data, dataset_name) {
  
  cat("\n=====================================\n")
  cat("Statistics for", dataset_name, "\n")
  cat("=====================================\n")
  
  numeric_cols <- data[, sapply(data, is.numeric)]
  
  for (col in names(numeric_cols)) {
    
    cat("\nColumn:", col, "\n")
    x <- numeric_cols[[col]]
    
    cat("Mean:", mean(x), "\n")
    cat("Median:", median(x), "\n")
    cat("Mode:", mode_function(x), "\n")
    cat("Standard Deviation:", sd(x), "\n")
    cat("Variance:", var(x), "\n")
    cat("Minimum:", min(x), "\n")
    cat("Maximum:", max(x), "\n")
    cat("Range:", diff(range(x)), "\n")
    cat("Quartiles:\n")
    print(quantile(x))
    cat("IQR:", IQR(x), "\n")
    
    cv <- (sd(x) / mean(x)) * 100
    cat("Coefficient of Variation:", cv, "%\n")
  }
}

# -----------------------------
# 5. Calculate Statistics for All 3 Datasets
# -----------------------------

calculate_statistics(data1, "Dataset 1")
calculate_statistics(data2, "Dataset 2")
calculate_statistics(data3, "Dataset 3")

# -----------------------------
# 6. Comparison Example (maxtempC)
# -----------------------------

comparison <- data.frame(
  Dataset = c("Dataset1", "Dataset2", "Dataset3"),
  Mean_MaxTemp = c(
    mean(data1$maxtempC),
    mean(data2$maxtempC),
    mean(data3$maxtempC)
  ),
  SD_MaxTemp = c(
    sd(data1$maxtempC),
    sd(data2$maxtempC),
    sd(data3$maxtempC)
  ),
  CV_MaxTemp = c(
    (sd(data1$maxtempC)/mean(data1$maxtempC))*100,
    (sd(data2$maxtempC)/mean(data2$maxtempC))*100,
    (sd(data3$maxtempC)/mean(data3$maxtempC))*100
  )
)

cat("\n\nComparison Table (Max Temperature):\n")
print(comparison)

############################################################
# END OF PROGRAM
############################################################