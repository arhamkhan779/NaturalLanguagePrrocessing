import numpy as np

class Normalize:
    def __init__(self,Contextual_Embeddings:np.array):
        self.Contextual_Embeddings=Contextual_Embeddings
        self.Beta = np.array([0.0,0.0,0.0,0.0])
        self.Gamma= np.array([1.0,1.0,1.0,1.0])
        self.Epsilon= 1e-5
    
    def Apply(self)-> np.array:
        try:
            mean=np.mean(self.Contextual_Embeddings)
            variance=np.var(self.Contextual_Embeddings)
            std_dev=np.sqrt((variance+self.Epsilon))
            
            normalize_contextual_embeddings=(self.Contextual_Embeddings-mean)/std_dev
            scale_shifted_embeddings= self.Gamma*normalize_contextual_embeddings+self.Beta 
            return scale_shifted_embeddings             
        except Exception as e:
            print(f"The Exception Lies as -> {e}")
    

    def positional_encoding(self):
        seq_length = len(self.Contextual_Embeddings)  # Number of tokens (positions)
        embedding_dim = len(self.Contextual_Embeddings)  # Embedding dimension matches input vector length
    
        # Initialize positional encoding matrix
        positional_encoding = np.zeros((seq_length, embedding_dim))
    
        # Position indices (0 to seq_length-1)
        positions = np.arange(seq_length)[:, np.newaxis]  # Shape: (seq_length, 1)
    
        # Dimension indices (0 to embedding_dim-1)
        dimensions = np.arange(embedding_dim)[np.newaxis, :]  # Shape: (1, embedding_dim)
    
        # Calculate the angles using the formula pos / 10000^(2i/d_model)
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / embedding_dim)
        angles = positions * angle_rates  # Broadcasting: (seq_length, embedding_dim)
    
        # Apply sin to even indices and cos to odd indices
        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Even indices
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Odd indices
    
        return positional_encoding



    



if __name__ == "__main__":
    obj=Normalize([2.0,4.0,6.0,8.0])
    print(obj.Apply())
    print(obj.positional_encoding())

