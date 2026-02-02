print("Hello R")

5 + 3      # Addition
10 - 4     # Subtraction
6 * 7      # Multiplication
20 / 5     # Division
10 %% 3    # Modulus
2 ^ 3      # Power

help(mean)
?mean

example(mean)

help.search("matrix")

x <- 10
y <- "R"
z <- TRUE

class(x)
class(y)
class(z)

v <- c(10, 20, 30, 40)
v

v + 5
v * 2
sum(v)
mean(v)

v[1]        # First element
v[2:4]      # Range
v[-1]       # Remove first element

v > 20
v[v > 30]

m <- matrix(1:9, nrow = 3, ncol = 3)
m

m + 5
m * 2
t(m)        # Transpose

a <- c(1, 2, 3)
b <- c(4, 5, 6)

a + b
a * b

m1 <- matrix(1:4, 2, 2)
m2 <- matrix(5:8, 2, 2)

m1 %*% m2


lst <- list(
  name = "Sarvesh",
  age = 20,
  marks = c(80, 85, 90)
)

lst
lst$name

df <- data.frame(
  Name = c("A", "B", "C"),
  Age = c(18, 19, 20),
  Marks = c(75, 82, 90)
)

df

df$Name
df[1, ]     # First row
df[, 2]     # Second column
