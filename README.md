---
author: Min Kang
date: 2023-06-13
output: md_document
title: Predicting Restaurant's Yelp Star Rating in the U.S.
---

`{r setup, include=FALSE} knitr::opts_chunk$set(echo = TRUE)`

------------------------------------------------------------------------

## Introduction

This project aims to predict the average number of stars a food
establishment in the U.S. will earn on Yelp.

### About our Data Set

According to [Yelp's website](https://www.yelp.com/dataset), the Yelp
Data Set contains a subset of data collected from businesses, reviews,
and users for public use.
[Documentation](https://www.yelp.com/dataset/documentation/main) reveals
that the data set is split into multiple JSON objects: `business.json`,
`review.json`, `user.json`, `checkin.json`, `tip.json`, and
`photo.json`. For this project, we'll only have to work with
`business.json`.

### Goals

There are many factors that can influence the ratings a restaurant
earns. Ideally, I'd like the ratings of a restaurant to prioritize
capturing the quality of the food they serve. However, biases,
preferences for certain services over others, cultural differences, and
many other such opinions people carry sway the results of an
establishment's ratings. The Yelp Data Set fortunately carries meta data
on restaurants that allow us to capture some of the qualities of a
restaurant aside from their food. Our goal is to see if we can
accurately predict the ratings a restaurant will receive using such
information. Specifically, this project will focus on establishments
that sell prepared food and drinks. We'll be excluding establishments
that primarily sell alcohol from this definition. Defining this right
now will help up later in choosing which businesses to include or
exclude from our model.

## Loading Packages and Data

We'll begin by loading all the necessary packages from our library.
We'll also be sure to specify that our code prioritizes the use of
tidymodel packages as we are taking advantage of the tidy workflow.

\`\`\`{r message = FALSE, warning = FALSE} \# Load packages

library(dplyr) library(ggplot2) library(glmnet) library(janitor)
library(jsonlite) library(magrittr) library(naniar) library(ranger)
library(readr) library(rsample) library(textrecipes) library(tidyverse)
library(tidymodels) library(vip) library(xgboost)

tidymodels_prefer()


    The yelp JSON objects are initially compiled into one `.tar` file. We'll have to unravel it to proceed. From there, we'll load `yelp_academic_dataset.business.json` as a data frame called `business` to start looking into our data.

    ```{r message = FALSE, warning = FALSE}
    untar("yelp_dataset.tar")

`{r message = FALSE, warning = FALSE} business <- jsonlite::stream_in(file("yelp_academic_dataset_business.json"))`

## Exploratory Data Analyses/Management

One of `business`'s features is `attributes`. This feature contains
valuable information such as the restaurant's price range, takeout
information, accessibility options, and more. However, not all of these
attributes are necessarily useful for us, such as `HairSpecializesIn`.
In order to make our modeling process simpler to manage and manipulate,
we'll flatten `attributes` so that every attribute becomes its own
feature.

\`\`\`{r message = FALSE, warning = FALSE} \# Flatten the nested JSON
structure within the attributes column attributes_df \<-
jsonlite::flatten(business\$attributes)

# Add the flattened attributes as individual columns in the business data frame

business \<- business %\>% select(-attributes) %\>%
bind_cols(attributes_df)


    ### Feature Selection

    After flattening `attributes`, we're left with 52 features and 150346 observations.

    ```{r}
    dim(business)

This leaves us with at least two issues we'll have to handle to move
forward.

First, many of these features are irrelevant to our project, like
`HairSpecializesIn`; some of these features have enormous amounts of
missing data, like `BusinessAccepsBitcoin`; and a few of these features
are simply too granular for our model, like `address`. Thus, I've chosen
to keep features that at a glance are relevant to food establishments,
appear to contain significant amounts of data, and are macroscopic
enough that they may have significant impact as predictors.

``` {r}
business_clean <- c("name", "city", "is_open", "address", "review_count", "postal_code", "latitude", "longitude", "hours", "BikeParking", "CoatCheck", "BusinessParking", "HappyHour", "HasTV", "GoodForKids", "RestaurantsAttire", "Ambience", "RestaurantsTableService", "RestaurantsGoodForGroups", "NoiseLevel", "GoodForMeal", "BusinessAcceptsBitcoin", "Smoking", "Music", "GoodForDancing", "AcceptsInsurance", "BestNights", "BYOB", "Corkage", "BYOBCorkage", "HairSpecializesIn", "Open24Hours", "RestaurantsCounterService", "AgesAllowed", "DietaryRestrictions", "DriveThru", "Alcohol", "Caters", "DogsAllowed", "ByAppointmentOnly", "WheelchairAccessible")

yelp <- business[, !(names(business) %in% business_clean)]
```

Second, many of these observations do not belong to businesses that
serve non-alcoholic prepared food or drinks. These businesses can range
from pet care stores, wineries, to automotive services. Luckily, there
exists a feature called `categories`. According to [this official Yelp
blog
post](https://blog.yelp.com/businesses/yelp_category_list/#h-restaurants),
categories are labels businesses can pick from to help direct users to
their establishment on Yelp. There are over 1,500 different categories
to pick from, and businesses are highly encouraged to pick as specific
categories as possible. This blog post shows major category groupings
such as Active Life, Education, etc., as well as all the possible sub
categories to pick from. From this list there are two major categories
our project will be concerned with, Food and Restaurants. However, these
category lists are not always clean. For example, Food contained
backshop as a category. There were also all the alcoholic categories to
deal with as well. Thus, I filtered through all the categories and
collected all the ones that fit under our project criteria. Finally, I
stored all the updated information under `yelp`.

``` {r}
restaurant_categories <- c("Acai Bowls", "Bagels", "Bakeries", "Bento", "Bubble Tea", "Chimney Cakes", "Churros", "Coffee & Tea", "Cupcakes", "Delicatessen", "Desserts", "Donairs", "Donuts", "Empanadas", "Food Trucks", "Friterie", "Gelato", "Hawker Centre", "Honey", "Ice Cream & Frozen Yogurt", "Internet Cafes", "Japanese Sweets", "Taiyaki", "Juice Bars & Smoothies", "Kombucha", "Milkshake Bars", "Nasi Lemak", "Panzerotti", "Parent Cafes", "Patisserie/Cake Shop", "Piadina", "Poke", "Pretzels", "Salumerie", "Shaved Ice", "Shaved Snow", "Smokehouse", "Specialty Food", "Sugar Shacks", "Tea Rooms", "Torshi", "Tortillas", "Zapiekanka", "Afghan", "African", "American (New)", "American (Traditional)", "Andulusian", "Arabian", "Argentine", "Armenian", "Asian Fusion", "Asturian", "Australian", "Austrian", "Baguettes", "Bangladeshi", "Barbeque", "Basque", "Bavarian", "Beisl", "Belfian", "Bistros", "Black Sea", "Brasseries", "Brazillian", "Brazilian Empanadas", "Central Brazillian", "Northeastern Brazillian", "Breakfast & Brunch", "British", "Buffets", "Bulgarian", "Burgers", "Burmese", "Cafes", "Cajun/Creole", "Cambodian", "Canadian (New)", "Caribbean", "Catalan", "Cheesesteaks", "Chicken Shop", "Chicken Wings", "Chilean", "Chinese", "Comfort Food", "Corsican", "Creperies", "Cuban", "Curry Sausage", "Cypriot", "Czech", "Czech/Slovakian", "Danish", "Delis", "Diners", "Dinner Theater", "Dumplings", "Eastern European", "Eritrean", "Ethiopean", "Fast Food", "Filipino", "Fishbroetchen", "Fish & Chips", "Flatbread", "Fondue", "Freiduria", "French", "French Southwest", "Galician", "Gastropubs", "Georgian", "German", "Giblets", "Greek", "Guamanian", "Halal", "Hawaiin", "Heuriger", "Himalayan\\Nepalese", "Honduran", "Hong Kong Style Cafe", "Hot Dogs", "Hot Pot", "Hungarian", "Iberian", "Indian", "Indonesian", "Irish", "Israeli", "Italian", "Japanese", "Jewish", "Kebab", "Kopitiam", "Korean", "Kosher", "Kurdish", "Laos", "Laotian", "Latin American", "Lyonnais", "Malaysian", "Meatballs", "Mediterranean", "Mexican", "Middle Eastern", "Milk Bars", "Modern Australian", "Modern European", "Mongolian", "Moroccan", "New Mexican Cuisine", "New Zealand", "Nicaraguan", "Nikkei", "Norcinerie", "Open Sandwhiches", "Oriental", "Pakistani", "Pan Asian", "Parma", "Persian/Iranian", "Peruvian", "Pita", "Pizza", "Polish", "Polynesian", "Portugese", "Poutineries", "Rice", "Romanian", "Rotisserie Chicken", "Russian", "Salad", "Sandwhiches", "Scandinavian", "Schnitzel", "Scottish", "Seafood", "Serbo Croatian", "Singaporean", "Slovakian", "Somali", "Soul Food", "Southern", "Spanish", "Sri Lankan", "Steakhouses", "Supper Clubs", "Sushi Bars", "Swabian", "Swedish", "Swiss Food", "Syrian", "Tabernas", "Taiwanese", "Tapas Bars", "Tapas/Small Plates", "Tavola Calda", "Tex-Mex", "Thai", "Traditional Norwegian", "Traditional Swedish", "Trattoerie", "Turkish", "Ukrainian", "Uzbek", "Vegan", "Vegetarian", "Venison", "Vietnamese", "Waffles", "Wok", "Wraps", "Yusoslav")

yelp$is_restaurant <- grepl(paste(restaurant_categories, collapse = "|"), yelp$categories, ignore.case = TRUE)

yelp <- yelp[yelp$is_restaurant, ]
```

It wouldn't be useful to simply have all the filtered categories exist
as predictors for our model. The sheer quantity of them would make our
model run inefficiently, and their granularity would make them poor
predictors. Thus, I chose 7 different food/drink features to place all
of these categories under. These boolean features are African, Asian,
Cafes/Desserts, European, Latin American, Middle Eastern, and North
American. I went through all of the filtered categories and hand placed
them under the appropriate feature. Now instead of listing the
restaurant's category, the data frame shows which major restaurant group
they do or do not belong to.

``` {r}
# Define the restaurant groups
restaurant_groups <- c(
  "African", 
  "Asian", 
  "Cafes/Desserts", 
  "European", 
  "Latin American",
  "Middle Eastern", 
  "North American"
)

# Create an empty list to store the category mapping
group_mapping <- list()

# Define the category mapping for each group
group_mapping$African <- c("Afghan", "African", "Eritrean", "Ethiopean", "Nicaraguan", "Somali")
group_mapping$Asian <- c("Bento", "Hawker Centre", "Taiyaki", "Nasi Lemak", "Poke", "Asian Fusion", "Bangladeshi", "Burmese", "Cambodian", "Chinese", "Dumplings", "Filipino", "Guamanian", "Himalayan\\Nepalese", "Hong Kong Style Cafe", "Hot Pot", "Indian", "Indonesian", "Japanese", "Kopitiam", "Korean", "Laos", "Laotian", "Malaysian", "Mongolian", "Nikkei", "Oriental", "Pan Asian", "Polynesian", "Rice", "Singaporean", "Sri Lankan", "Sushi Bars", "Taiwanese", "Thai", "Vietnamese", "Wok")
group_mapping$`Cafes/Desserts` <- c("Acai Bowls", "Bagels", "Bakeries", "Bubble Tea", "Chimney Cakes", "Churros", "Coffee & Tea", "Cupcakes", "Desserts", "Donuts", "Gelato", "Honey", "Ice Cream & Frozen Yogurt", "Internet Cafes", "Japanese Sweets", "Juice Bars & Smoothies", "Kombucha", "Milkshake Bars", "Parent Cafes", "Patisserie/Cake Shop", "Shaved Ice", "Shaved Snow", "Sugar Shacks", "Tea Rooms", "Breakfast & Brunch", "Cafes", "Cheesesteaks", "Creperies", "Milk Bars")
group_mapping$European <- c("Delicatessen", "Friterie", "Panzerotti", "Piadina", "Pretzels","Salumerie", "Torshi", "Zapiekanka", "Andulusian", "Armenian", "Asturian", "Australian", "Austrian", "Baguettes", "Basque", "Bavarian", "Beisl", "Belfian", "Bistros", "Black Sea", "Brasseries", "British", "Bulgarian", "Catalan", "Corsican", "Curry Sausage", "Cypriot", "Czech", "Czech/Slovakian", "Danish", "Eastern European", "Fishbroetchen", "Fish & Chips", "Fondue", "Freiduria", "French", "French Southwest", "Galician", "Gastropubs", "Georgian", "German", "Giblets", "Greek",
  "Heuriger", "Hungarian", "Iberian", "Irish", "Italian", "Lyonnais", "Meatballs", "Mediterranean", "Modern Australian", "Modern European", "New Zealand", "Norcinerie", "Parma", "Polish", "Portugese", "Romanian", "Russian", "Scandinavian", "Schnitzel", "Scottish", "Serbo Croatian", "Slovakian", "Spanish", "Swabian", "Swedish", "Swiss Food", "Tabernas", "Tapas Bars", "Tapas/Small Plates", "Tavola Calda", "Traditional Norwegian", "Traditional Swedish", "Trattoerie", "Turkish", "Ukrainian", "Uzbeck", "Venison", "Yusoslav") 
group_mapping$`Latin American` <- c("Empanadas", "Tortillas", "Argentine", "Brazillian", "Brazilian Empanadas", "Central Brazillian", "Northeastern Brazillian", "Caribbean", "Chilean", "Cuban", "Latin American", "Mexican", "New Mexican Cuisine", "Peruvian")
group_mapping$`Middle Eastern` <- c("Arabian", "Flatbread", "Halal", "Israeli", "Jewish", "Kebab", "Kosher", "Kurdish", "Moroccan", "Pakistani", "Persian/Iranian", "Pita", "Syrian")
group_mapping$`North American` <- c("Donairs", "Food Trucks", "Smokehouse", "Specialty Food", "American (New)", "American (Traditional)", "Barbeque", "Buffets", "Burgers", "Cajun/Creole", "Canadian (New)", "Chicken Shop", "Chicken Wings", "Comfort Food", "Delis", "Diners", "Dinner Theater", "Fast Food", "Hawaiin", "Honduran", "Hot Dogs", "Open Sandwhiches", "Pizza", "Poutineries", "Rotisserie Chicken", "Salad", "Sandwhiches", "Seafood", "Soul Food", "Southern", "Steakhouses", "Supper Clubs", "Tex-Mex", "Vegan", "Vegetarian", "Waffles", "Wraps")

# Create new columns for each category
for (group in restaurant_groups) {
  categories <- group_mapping[[group]]
  yelp[[group]] <- grepl(paste(categories, collapse = "|"), yelp$categories, ignore.case = TRUE)
}

# Remove the 'categories' and 'is_restaurant' columns
yelp <- yelp[, !(names(yelp) %in% c("categories", "is_restaurant"))]
```

### Normalizing values

Due to the nature of the Yelp Data Set, all of our predictors are
nominal. Many of them are boolean in nature. However, the values they
take on to represent the same things can differ from predictor to
predictor. Many of these predictors show `none`, `u'no'`, and `False`
just to represent the same outcome. So, we apply a function to normalize
value outcomes to prevent our model from misinterpreting the data.

``` {r}
# Get unique values of the predictors
BusinessAcceptsCreditCards_values <- unique(yelp$BusinessAcceptsCreditCards)
RestaurantReservations_values <- unique(yelp$RestaurantsReservations)
WiFi_values <- unique(yelp$WiFi)
RestaurantsTakeOut_values <- unique(yelp$RestaurantsTakeOut) 
RestaurantsDelivery_values <- unique(yelp$RestaurantsDelivery)
OutdoorSeating_values <- unique(yelp$OutdoorSeating)

# Print the unique values
print(BusinessAcceptsCreditCards_values)
print(RestaurantReservations_values)
print(WiFi_values)
print(RestaurantsTakeOut_values)
print(RestaurantsDelivery_values)
print(OutdoorSeating_values)

# Define a function to normalize predictors
normalize <- function(value) {
    if (value %in% c("u'free'", "'free'")) {
    return("Free")
  } else if (value %in% c("u'paid'", "'paid'")) {
    return("Paid")
  } else if (value %in% c("None", "False", "u'no'", "no", "FALSE")) {
    return("false")
  } else if (value %in% c("True", "TRUE")) {
    return("true")
  } else {
    return(value)
  }
}

# Apply the normalization function
yelp$BusinessAcceptsCreditCards <- sapply(yelp$BusinessAcceptsCreditCards, normalize)
print(BusinessAcceptsCreditCards_values)

yelp$RestaurantsReservations <- sapply(yelp$RestaurantsReservations, normalize)
print(RestaurantReservations_values)

yelp$WiFi <- sapply(yelp$WiFi, normalize)
print(WiFi_values)

yelp$RestaurantsTakeOut <- sapply(yelp$RestaurantsTakeOut, normalize)
print(RestaurantsTakeOut_values)

yelp$RestaurantsDelivery <- sapply(yelp$RestaurantsDelivery, normalize)
print(RestaurantsDelivery_values)

yelp$OutdoorSeating <- sapply(yelp$OutdoorSeating, normalize)
print(OutdoorSeating_values)

yelp$African <- sapply(yelp$African, normalize)
yelp$Asian <- sapply(yelp$Asian, normalize)
yelp$`Cafes/Desserts` <- sapply(yelp$`Cafes/Desserts`, normalize)
yelp$European <- sapply(yelp$European, normalize)
yelp$`Latin American` <- sapply(yelp$`Latin American`, normalize)
yelp$`Middle Eastern` <- sapply(yelp$`Middle Eastern`, normalize)
yelp$`North American` <- sapply(yelp$`North American`, normalize)
```

### Converting Factors

Since all of our predictors are discrete, with only our outcome being
continuous, we factorize all of our predictors.

``` {r}
yelp$business_id <- as.factor(yelp$business_id)
yelp$state <- as.factor(yelp$state)
yelp$BusinessAcceptsCreditCards <- as.factor(yelp$BusinessAcceptsCreditCards)
yelp$RestaurantsPriceRange2 <- as.factor(yelp$RestaurantsPriceRange2)
yelp$RestaurantsTakeOut <- as.factor(yelp$RestaurantsTakeOut)
yelp$RestaurantsDelivery <- as.factor(yelp$RestaurantsDelivery)
yelp$WiFi <- as.factor(yelp$WiFi)
yelp$OutdoorSeating <- as.factor(yelp$OutdoorSeating)
yelp$RestaurantsReservations <- as.factor(yelp$RestaurantsReservations)
yelp$African <- as.factor(yelp$African)
yelp$Asian <- as.factor(yelp$Asian)
yelp$`Cafes/Desserts` <- as.factor(yelp$`Cafes/Desserts`)
yelp$European <- as.factor(yelp$European)
yelp$`Latin American` <- as.factor(yelp$`Latin American`)
yelp$`Middle Eastern` <- as.factor(yelp$`Middle Eastern`)
yelp$`North American` <- as.factor(yelp$`North American`)

head(yelp)
```

### Missing Data

A necessary step to move forward is to look for missing data. We can
observe that `WiFi` and `RestaurantReservations` are the two features
left with the highest percentage of missing data. Since both of these
features are not expected of all food/drink establishments, e.g.) it's
unlikely any boba store would require reservations, we can simply omit
these features from the model.

``` {r}
yelp %>% vis_miss(warn_large_data = FALSE, sort_miss = TRUE)
```

The rest of our missing data is now small enough that imputation becomes
a more viable solution. Since the predictors with missing data are
nominal, we'll choose to impute with mode. This will be done later in
our recipe creation.

``` {r}
yelp <- yelp[, !(names(yelp) %in% c("WiFi", "RestaurantsReservations"))]
yelp %>% vis_miss(warn_large_data = FALSE, sort_miss = TRUE)
```

### Visual Data Exploration

It may be helpful to see the distribution of our outcome. We can see
here that the mean distribution of stars is left skewed. This suggests
that there exists a factor that encourages people to rank restaurants
more favorably. Hopefully, our model captures this.

``` {r}
ggplot(yelp, aes(stars)) + 
  geom_bar(fill = rgb(255, 26, 26, maxColorValue = 255)) +
  labs(title = "Distribution of Stars")
```

We should also see what the distribution of restaurants in the U.S. are
according to our definitions. Unsurprisingly, a majority of the
restaurants are classified as North American, this is despite European
containing significantly more categories. It also appears there aren't
many Middle Eastern or African restaurants in the U.S.

``` {r}
# Reshape the data frame to long format
yelp_long <- tidyr::pivot_longer(yelp, cols = c(African, Asian, `Cafes/Desserts`, European, `Latin American`, `Middle Eastern`, `North American`))

# Create a bar plot
ggplot(yelp_long, aes(x = name, fill = value)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("#FFFFFF", "#FF1A1A"), labels = c("FALSE", "TRUE")) +
  labs(x = "Predictor", y = "Proportion", title = "Distribution of TRUE values among Predictors") +
  theme(axis.text = element_text(size = 8))
```

## Model Preperation

Before we begin creating our model, we'll set the seed for the sake of
reproducibility.

``` {r}
set.seed(92)
```

We'll now split our data frame into a training and testing set. The
training set will serve to build our models. We'll then assess the
models based off their RMSE, root mean squared error, scores. Our
testing set will serve to compare the results of our models on a portion
of data it has never seen before. This will allow us to see if we've
over or under-fitted our model. I've chosen to use a 70% of the data for
the training set and 30% of the data for the testing set. The objective
here is to strike a good balance of giving our models as much training
data as possible while still having enough data left for our testing
test to make meaningful comparisons to. Having our testing set being
proportionately smaller will allow us to observe any over or
under-fitting that may occur. We'll stratify our data on our outcome
`stars`. Thus, we can be assured there is an equal proportion of our
outcome variable in both the testing and training set. We can also
observe if our data was split as intended.

``` {r}
yelp_split <- initial_split(yelp, prop = 0.70, strata = stars)
yelp_train <- training(yelp_split)
yelp_test <- testing(yelp_split)

nrow(yelp_train)/nrow(yelp)
nrow(yelp_test)/nrow(yelp)
```

### Recipe Formulation

Since we've done so much cleaning on our data, we'll be able to use all
of our features: `state`, `BusinessAcceptsCreditCards`,
`RestaurantPriceRange2`, `RestaurantsTakeOut`, `RestaurantsDelivery`,
`WiFi`, `OutdoorSeating`, `RestaurantsReservations`, `African`, `Asian`,
`European`, `Latin American`, `Middle Eastern`, and `North American` as
predictors in our recipe. That is, with an exception. `business_id` will
be our one feature that will neither be a predictor nor an outcome. This
is just to traceback to identify any standout businesses later in case
we desire to. We'll also choose to use the same recipe for all of our
models.

Our recipe also requires several other crucial steps. `step_zv()`
removes any zero variance predictors. That is, it'll remove any
predictors that have no influence on the outcome of our models. We'll
also impute the predictors with missing data from earlier with
`step_impute_mode()`. Since our predictors are categorical,
`step_dummy()` will convert our predictors into indicator variables for
our models to be able to interpret.

``` {r}
yelp_recipe <- recipe(yelp_train) %>%
  
  update_role(all_nominal(), new_role = "id") %>%
  
  update_role(stars, new_role = "outcome") %>%
  
  update_role(state, BusinessAcceptsCreditCards, RestaurantsTakeOut, RestaurantsDelivery, OutdoorSeating, African, Asian, `Cafes/Desserts`, European, `Latin American`, `Middle Eastern`, `North American`, new_role = "predictor") %>%
  
  step_zv(all_nominal_predictors()) %>%
  
  step_impute_mode(all_nominal(), -all_outcomes()) %>%
  
  step_dummy(state, BusinessAcceptsCreditCards, RestaurantsTakeOut, RestaurantsDelivery, OutdoorSeating, African, Asian, `Cafes/Desserts`, European, `Latin American`, `Middle Eastern`, `North American`) %>%
  
  step_center(all_predictors()) %>%
  step_scale(all_predictors())
```

### K-Fold Cross Validation

Our models will take advantage of k-fold cross validation. That is, our
data split into 9 different folds. Our models will be trained on 8 of
the 9 folds and tested against the remaining fold. This will occur 9
times, and we'll take the average metrics across the 9 runs to measure
our model performance. Similar to how we stratified our data on `stars`
during the split earlier, we'll do the same when creating these folds.
Cross validation is a useful tool in observing the bias-variance trade
offs of our model as well as increasing our model's overall robustness.

``` {r}
yelp_folds <- vfold_cv(yelp_train, v = 9, strata = stars)
```

## Model Building

We'll now design the models for our experiment. Since our outcome is
continuous, we'll be building regression models. We'll also choose RMSE
as our metric for evaluation when picking our best models. That is,
we'll pick the model with the lowest overall RMSE. The models of choice
are linear regression, ridge regression, lasso regression, k-nearest
neighbors, and xgboost.

``` {r}
linear <- linear_reg() %>%
  set_engine("lm")
  
ridge <- linear_reg(mixture = 0, penalty = tune()) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

lasso <- linear_reg(mixture = 1, penalty = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet")

knn <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("regression") %>%
  set_engine("kknn")

bt <- boost_tree(trees = tune(), learn_rate = tune(), min_n = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")
```

Here, we'll design the workflows for our models using our models and
recipe.

``` {r}
linear_wf <- workflow() %>%
  add_model(linear) %>%
  add_recipe(yelp_recipe)

ridge_wf <- workflow() %>%
  add_model(ridge) %>%
  add_recipe(yelp_recipe)

lasso_wf <- workflow() %>%
  add_model(lasso) %>%
  add_recipe(yelp_recipe)

knn_wf <- workflow() %>%
  add_model(knn) %>%
  add_recipe(yelp_recipe)

bt_wf <- workflow() %>%
  add_model(bt) %>%
  add_recipe(yelp_recipe)
```

Here we'll create tuning grids to set the parameters for model tuning.

``` {r}
penalty_grid <- grid_regular(penalty(range = c(-5, 5)), levels = 50)

neighbors_grid <- grid_regular(neighbors(range = c(1, 15)), levels = 5)

bt_grid <- grid_regular(trees(range = c(5, 200)), learn_rate(range = c(0.01, 0.1), trans = identity_trans()), min_n(range = c(40, 60)), levels = 5)
```

We'll now tune the models specifying our workflows, folds, and tuning
grids.

\`\`\`{r message = FALSE, warning = FALSE} \# linear regression does not
require tuning linear_fit \<- fit_resamples(linear_wf, resamples =
yelp_folds)

ridge_tune \<- tune_grid(ridge_wf, resamples = yelp_folds, grid =
penalty_grid)

lasso_tune \<- tune_grid(lasso_wf, resamples = yelp_folds, grid =
penalty_grid)

knn_tune \<- tune_grid(knn_wf, resamples = yelp_folds, grid =
neighbors_grid)

bt_tune \<- tune_grid(bt_wf, resamples = yelp_folds, grid = bt_grid)


    We'll now read and write these models into our project directory to save computation time.

    ```{r}
    write_rds(linear_fit, file = "tuned_models/linear.rds")
    write_rds(ridge_tune, file = "tuned_models/ridge.rds")
    write_rds(lasso_tune, file = "tuned_models/lasso.rds")
    write_rds(knn_tune, file = "tuned_models/knn.rds")
    write_rds(bt_tune, file = "tuned_models/bt.rds")

``` {r}
linear_fit <- read_rds(file = "tuned_models/linear.rds")
ridge_tune <- read_rds(file = "tuned_models/ridge.rds")
lasso_tune <- read_rds(file = "tuned_models/lasso.rds")
knn_tune <- read_rds(file = "tuned_models/knn.rds")
bt_tune <- read_rds(file = "tuned_models/bt.rds")
```

## Model Results

\`\`\`{r message = FALSE, warning = FALSE} linear_rmse \<-
show_best(linear_fit, metrics = "rmse")\[1,\]

ridge_rmse \<- show_best(ridge_tune, metrics = "rmse")\[1,\]

lasso_rmse \<- show_best(lasso_tune, metrics = "rmse")\[1,\]

knn_rmse \<- show_best(knn_tune, metrics = "rmse")\[1,\]

bt_rmse \<- show_best(bt_tune, metrics = "rmse")\[1,\]


    We can see that our boosted trees model performed the best. Ridge regression, lasso regression, and linear regression all performed similarly to one another, and was also only marginally worse than boosted trees. However, k-nearest neighbors clearly performed the worst. This could be due to the imbalance of restaurants in the U.S..

    ```{r}
    final_tibble <- tibble(Model = c("Linear Regression", "Ridge Regression", "Lasso Regression","K Nearest Neighbors", "Boosted Trees"), RMSE = c(linear_rmse$mean, ridge_rmse$mean, lasso_rmse$mean, knn_rmse$mean, bt_rmse$mean))

    # Arranging by lowest RMSE
    final_tibble <- final_tibble %>% 
      arrange(RMSE)

    print(final_tibble)

### Best Model Visualization

The xgboost model performance across different levels can be seen below.
Increasing the learning rate past 0.055 does not seem to significantly
improve the model performance, however it can be generalized that
increasing learning rate improves model performance. It is also worth
note that after 50 trees our model performance appears to plateau.

``` {r}
autoplot(bt_tune, metric = "rmse")
```

### Model Testing

First, we select the best xgboost model to fit onto the entirety of our
training data.

\`\`\`{r message = FALSE, warning = FALSE} best_bt_train \<-
select_best(bt_tune, metric = "rmse") bt_final_workflow_train \<-
finalize_workflow(bt_wf, best_bt_train) bt_final_fit_train \<-
fit(bt_final_workflow_train, data = yelp_train)

write_rds(bt_final_fit_train, file = "bt_final/bt_final_train.rds")


    Then, we test the model on our testing set.

    ```{r}
    yelp_tibble <- predict(bt_final_fit_train, new_data = yelp_test %>% select(-stars))
    yelp_tibble <- bind_cols(yelp_tibble, yelp_test %>% select(stars))

The RMSE of the testing set, 0.8159, performed slightly worse than the
training set, 0.8125. It is important to note that the values RMSE take
on are in relation to the scale of the outcome. Since Yelp rates from 0
to 5 stars, an RMSE of 0.8159 is not an ideal score. Our model is only
able to account for some of the variation in the account.

``` {r}
yelp_metric <- metric_set(rmse)

yelp_tibble_metrics <- yelp_metric(yelp_tibble, truth = stars, estimate = .pred)
yelp_tibble_metrics
```

The plot of our predicted vs.Â actual values is somewhat illuminating to
what is occurring. It seems our model likes to predict that just about
restaurant will like to sit somewhere close to the 3-4 star range. This
aligns fairly close with the distribution of stars we observed earlier.
However, the model seems to do a poor job of predicting when a
restaurant will receive a low rating.

``` {r}
yelp_tibble %>%
  ggplot(aes(x = .pred, y = stars)) +
  geom_point(alpha = 0.1, color = rgb(255, 26, 26, maxColorValue = 255)) +
  geom_abline(lty = 2, color = rgb(255, 26, 26, maxColorValue = 255)) +
  coord_obs_pred() +
  theme_light() +
  labs(title = "Predicted Values vs. Actual Values")
```

### Variable Importance Plot

Since xgboost was our best model, we're able to use a VIP, variable
important plot, to observe the most significant predictors in our
outcome. Whether or not a restaurant was `North American_true` is by far
our best predictor, but that could be due to the sheer number of
restaurants that were North American. Our second best predictor is
`RestaurantsDelivery_true`. This indicates that the most important
service a restaurant can offer for their ratings is if they deliver or
not.

``` {r}
bt_final_fit_train %>% 
  extract_fit_engine() %>% 
  vip(aesthetics = list(fill = rgb(255, 26, 26, maxColorValue = 255), color = "white"))
```

## Conclusion

After running our models, it seems it would be difficult to gauge using
Yelp meta data alone to predict the average stars a restaurant will
receive. After testing out several models, xgboost resulted in the
lowest RMSE compared to the others, but not by a lot. None of the models
had particularly strong RMSE scores either.

Model improvements could definitely be made with perhaps longer training
times or finer tuning, but I think the biggest losses, or in this case
gains, in RMSE was due to how much data was lost between all the
features we were forced to exclude. The Yelp Data Set also contains but
a subset of their data base, and I am unsure if this data was stratified
evenly across all features. Would it have been possible to obtain more
observations on restaurants that exist outside of the North American
category. There also could be significant sources of error in how I
chose to group restaurants, and even more error exists when thinking
about how well businesses chose to categorize themselves.

If I were to iterate on this project, I would look into analyzing
`review.json`. Perhaps sentiment analysis would be able to capture
reviewers thoughts more robustly than a star system ever could. It might
also be worth the effort to categorize users based off user meta data.
Perhaps the ratings a user gives is more dependent on the user than the
establishment they go to.

For the purposes of my research question, the outcome of this project,
although not perfectly concluded, is reassuring to observe. Since there
is no direct business meta data on the quality of food, the fact we
weren't able to strongly predict ratings accurately suggests a
possibility that food quality is an important feature of a restaurant's
rating that we failed to capture. Thus, for now, I can be rest assured
that perhaps I can dine happily at a highly rated Yelp restaurant.

## Sources

The data comes from the [Yelp
Dataset](https://www.yelp.com/dataset/documentation/main) and its
documentation can be found
[here](https://www.yelp.com/dataset/documentation/main)

The [official Yelp blog
post](https://blog.yelp.com/businesses/yelp_category_list/#h-restaurants)
contains all the information related to categories.

[Brand Palletes](https://brandpalettes.com/yelp-colors/) gave
information on Yelp colors for plot purposes.
