import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.distributions import Normal
class ImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(weights=Resnet18_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.fc = nn.Linear(512,256)

    def forward(self,x):

        x = self.feature_extractor(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x
    
class StateEncoder(nn.Module):


    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(18,128),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Linear(128,128),

            nn.LayerNorm(128),

            nn.GELU()

        )

    def forward(self,state):

        return self.network(state)

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(640,512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512,512),
            nn.LayerNorm(512),
            nn.GELU()
        )
    def forward( self,front_feature,
        down_feature,state_feature ):

        x = torch.cat((front_feature,
         down_feature,state_feature ), dim=1 )
        return self.network(x)
class TemporalMemory(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=512,
        num_layers=1,
        dropout=0.0
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(
        self,
        x,
        hidden_state=None,
        cell_state=None,
    ):

        if hidden_state is None or cell_state is None:

            output,(hidden_state,cell_state)=self.lstm(x)

        else:

            output,(hidden_state,cell_state)=self.lstm(
                x,
                (hidden_state,cell_state)
            )

        return output,hidden_state,cell_state

    def initialize_memory(self,batch_size,device):

        hidden_state=torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )

        cell_state=torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )

        return hidden_state,cell_state
class ActorHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128,3)
        )
    def forward(self,x):
        return self.network(x)
    
class CriticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.network(x)

class ActorCritic(nn.Module):
    def __init__(self):
        self.front_encoder = ImageEncoder()
        self.down_encoder = ImageEncoder()
        self.state_encoder = StateEncoder()
        self.fusion = FeatureFusion()
        self.memory = TemporalMemory()
        self.actor = ActorHead()
        self.critic = CriticHead()
        self.log_std = nn.Parameter(torch.zeros(3))
    def forward( self, state, 
                front_image=None,
                 down_image=None,
        hidden_state=None,
          cell_state=None):
        front_feature = self.front_encoder(
            front_image)#256 numbers
        down_feature = self.down_encoder(
            down_image)#256 numbers 
        state_feature = self.state_encoder(state)#128 numbers

        fused_feature = self.feature_fusion(
            front_feature,
            down_feature,
            state_feature)#512 numbers 
        fused_feature = fused_feature.unsqueeze(1)

        lstm_output, hidden_state, cell_state = self.temporal_memory(

            fused_feature,

            hidden_state,

            cell_state

        )


        lstm_output = lstm_output.squeeze(1)

        action_mean = self.actor(
            lstm_output
        )

        # -----------------------------------
        # Critic
        # -----------------------------------

        state_value = self.critic(
            lstm_output
        )

        return (

            action_mean,

            state_value,

            hidden_state,

            cell_state
        )
    def act(self,state,front_image=None,
        down_image=None, hidden_state=None, cell_state=None):
        action_mean, state_value, hidden_state, cell_state = self.forward(
            state,
            front_image,
            down_image,
            hidden_state,
            cell_state )

        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean,action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return (
            action,
            log_prob,
            state_value,
            hidden_state,
            cell_state  )
def evaluate(
    self,
    state,
    action,
    front_image=None,
    down_image=None,
    hidden_state=None,
    cell_state=None
):

    action_mean, state_value, _, _ = self.forward(

        state,

        front_image,

        down_image,

        hidden_state,

        cell_state

    )

    action_std = torch.exp(self.log_std)

    dist = Normal(
        action_mean,
        action_std
    )

    log_prob = dist.log_prob(action).sum(dim=-1)

    entropy = dist.entropy().sum(dim=-1)

    return (

        log_prob,

        entropy,

        state_value

    )