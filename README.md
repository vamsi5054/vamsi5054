from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate, train_test_split

# Sample data (user, item, rating)
data = [
    ('User1', 'Movie1', 5),
    ('User1', 'Movie2', 4),
    ('User2', 'Movie1', 3),
    ('User2', 'Movie3', 2),
    ('User3', 'Movie2', 5),
    # Add more data as needed
]

# Define the data structure for Surprise
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data, reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Use KNNBasic algorithm for collaborative filtering
sim_options = {
    'name': 'cosine',
    'user_based': False
}

model = KNNBasic(sim_options=sim_options)

# Train the model
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate the model
from surprise import accuracy

accuracy.rmse(predictions)

