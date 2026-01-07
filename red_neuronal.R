library(neuralnet)
library(caTools)
library(caret)

cereals <- read.csv("cereal.csv", stringsAsFactors = FALSE)
str(cereals)
summary(cereals)
cereals <- cereals[, sapply(cereals, is.numeric)] # Eliminar columnas no numericas
cereals <- na.omit(cereals) # Eliminar valores faltantes

# Normalizacion Min-Max
normalize <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

# Para ver los intervalos de cada clase
cereals_norm <- as.data.frame(lapply(cereals, normalize))
intervalos <- cut(
  cereals_norm$rating,
  breaks = 3
)
levels(intervalos)

cereals_norm$rating_class <- cut(
  cereals_norm$rating,
  breaks = 3,
  labels = c("Bajo", "Medio", "Alto")
)
cereals_norm$rating <- NULL
cereals_norm$rating_class <- as.factor(cereals_norm$rating_class)

set.seed(42)
split <- sample.split(cereals_norm$rating_class, SplitRatio = 0.60)
entrenamiento <- cereals_norm[split, ]
prueba <- cereals_norm[!split, ]
Y <- model.matrix(~ rating_class - 1, data = entrenamiento)
X <- entrenamiento[, !names(entrenamiento) %in% "rating_class"]
datos_nn <- data.frame(Y, X)

form <- as.formula(
  paste(
    paste(colnames(Y), collapse = " + "),
    "~",
    paste(colnames(X), collapse = " + ")
  )
)

red <- neuralnet(
  form,
  data = datos_nn,
  hidden = c(6, 4),
  learningrate = 0.2,
  threshold = 0.01,
  linear.output = FALSE
)

plot(red)
X_prueba <- prueba[, !names(prueba) %in% "rating_class"]
pred <- compute(red, X_prueba)
colnames(pred$net.result) <- colnames(Y)

clase_predicha <- apply(
  pred$net.result,
  1,
  function(x) colnames(Y)[which.max(x)]
)

clase_predicha <- gsub("rating_class", "", clase_predicha)
clase_predicha <- as.factor(clase_predicha)

confusion <- table(
  Real = prueba$rating_class,
  Predicho = clase_predicha
)

confusion
confusionMatrix(clase_predicha, prueba$rating_class)
# Convertir la matriz a data frame
conf_df <- as.data.frame(confusion)

# Grafica tipo heatmap
ggplot(conf_df, aes(x = Predicho, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Matriz de Confusion",
    x = "Clase Predicha",
    y = "Clase Real"
  ) +
  theme_minimal()
