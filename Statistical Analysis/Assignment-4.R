############################################################
# ASSIGNMENT 4 - GRAPHICAL REPRESENTATION (ONE DATASET)
############################################################

# ---- 1. Load Data ----
data <- read.csv("./Statistical Analysis/pune-1.csv", 
                 stringsAsFactors = FALSE)

cat("Structure of dataset:\n")
str(data)

cat("\nColumn Names:\n")
print(colnames(data))

############################################################
# ---- 2. Data Cleaning ----
############################################################

# Convert required numeric columns
numeric_columns <- c("maxtempC", "mintempC", 
                     "totalSnow_cm", "sunHour", 
                     "uvIndex", "moon_illumination")

for (col in numeric_columns) {
  if (col %in% names(data)) {
    data[[col]] <- as.numeric(gsub("[^0-9.-]", "", data[[col]]))
  }
}

# Remove missing values
data <- na.omit(data)

############################################################
# ---- 3. Frequency Distribution (Max Temp) ----
############################################################

cat("\nFrequency Distribution of Max Temperature:\n")
temp_freq <- table(cut(data$maxtempC, breaks = 5))
print(temp_freq)

############################################################
# ---- 4. Bar Plot (UV Index - Discrete Example) ----
############################################################

png("barplot_uvindex.png")
barplot(table(data$uvIndex),
        main = "Bar Plot of UV Index",
        xlab = "UV Index",
        ylab = "Frequency",
        col = "skyblue")
dev.off()

############################################################
# ---- 5. Pie Chart (UV Index) ----
############################################################

png("pie_uvindex.png")
pie(table(data$uvIndex),
    main = "Pie Chart of UV Index",
    col = rainbow(length(unique(data$uvIndex))))
dev.off()

############################################################
# ---- 6. Histogram (Continuous Data - Max Temp) ----
############################################################

png("histogram_maxtemp.png")
hist(data$maxtempC,
     main = "Histogram of Max Temperature",
     xlab = "Max Temperature (°C)",
     col = "lightgreen",
     border = "black")
dev.off()

############################################################
# ---- 7. Frequency Polygon (Max Temp) ----
############################################################

png("frequency_polygon_maxtemp.png")
h <- hist(data$maxtempC, plot = FALSE)
plot(h$mids, h$counts,
     type = "l",
     main = "Frequency Polygon of Max Temperature",
     xlab = "Max Temperature",
     ylab = "Frequency",
     col = "blue",
     lwd = 2)
dev.off()

############################################################
# ---- 8. Box-and-Whisker Plot ----
############################################################

png("boxplot_maxtemp.png")
boxplot(data$maxtempC,
        main = "Box Plot of Max Temperature",
        col = "orange")
dev.off()

############################################################
# ---- 9. Scatter Plot (Max Temp vs Min Temp) ----
############################################################

png("scatter_temp.png")
plot(data$maxtempC, data$mintempC,
     main = "Scatter Plot: Max Temp vs Min Temp",
     xlab = "Max Temperature",
     ylab = "Min Temperature",
     col = "red",
     pch = 19)
dev.off()

############################################################
# ---- 10. Stem-and-Leaf Plot ----
############################################################

cat("\nStem-and-Leaf Plot of Max Temperature:\n")
stem(data$maxtempC)

############################################################
# Continuous vs Discrete Explanation
############################################################

cat("\nContinuous Data Example: maxtempC, mintempC\n")
cat("Discrete Data Example: uvIndex\n")

############################################################
# END OF PROGRAM
############################################################