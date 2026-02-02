# if
marks <- 85

if (marks >= 90) {
  print("Grade A")
} else if (marks >= 75) {
  print("Grade B")
} else {
  print("Grade C")
}

# for
for (i in 1:5) {
  print(i)
}

# functions
check_even <- function(n) {
  if (n %% 2 == 0) {
    return("Even")
  } else {
    return("Odd")
  }
}

check_even(7)

data <- read.csv("temperature.csv")
head(data)
