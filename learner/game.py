import os
import pygame
import random
import sys

class GameState:
    def __init__(self, width=800, height=600, render_game=True):
        # Initialize pygame without audio to avoid ALSA errors in WSL2
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        pygame.init()
        self.screen_width = width
        self.screen_height = height
        self.render_game = render_game
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Box Containment Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.thirty_second_timer = pygame.time.get_ticks() + 30000  # 30 seconds from start
        
        # Game boundaries
        self.upper_bound = 50
        self.lower_bound = height - 50
        
        # Red box properties (random y movement within bounds)
        self.red_box = {
            'rect': pygame.Rect(width//2 - 15, height//2, 30, 30),
            'color': (255, 0, 0),
            'y_speed': 0,
            'change_counter': 0,
            'max_speed': 4  # Maximum speed for red box
        }
        
        # White container box properties
        self.white_box = {
            'rect': pygame.Rect(width//2 - 100, height//2, 200, 200),
            'color': (255, 255, 255),
            'y_speed': 0,
            'gravity': 0.3,
            'lift': -0.5,  # Upward acceleration when clicked
            'max_speed': 8
        }
        
        # Game state variables
        self.score = 0
        self.score_rate = 1  # Points per frame while contained
        self.contained = False
        self.font = pygame.font.SysFont(None, 36)
        self.clicking = False
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.clicking = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.clicking = False
    
    def update(self):
        # Update red box (random y movement within game bounds)
        self.red_box['change_counter'] += 1
        if self.red_box['change_counter'] > 30:  # Change direction every half second
            self.red_box['y_speed'] = random.uniform(-self.red_box['max_speed'], self.red_box['max_speed'])
            self.red_box['change_counter'] = 0
        
        self.red_box['rect'].y += self.red_box['y_speed']
        
        # Keep red box within game bounds (bounce off edges)
        if self.red_box['rect'].top < self.upper_bound:
            self.red_box['rect'].top = self.upper_bound
            self.red_box['y_speed'] *= -1
        if self.red_box['rect'].bottom > self.lower_bound:
            self.red_box['rect'].bottom = self.lower_bound
            self.red_box['y_speed'] *= -1
        
        # Update white box physics
        if self.clicking:
            # Apply lift while mouse is held
            self.white_box['y_speed'] += self.white_box['lift']
        else:
            # Apply gravity when not clicking
            self.white_box['y_speed'] += self.white_box['gravity']
        
        # Limit maximum speed
        self.white_box['y_speed'] = max(-self.white_box['max_speed'], 
                                       min(self.white_box['y_speed'], self.white_box['max_speed']))
        
        self.white_box['rect'].y += self.white_box['y_speed']
        
        # Keep white box within game bounds
        if self.white_box['rect'].top < self.upper_bound:
            self.white_box['rect'].top = self.upper_bound
            self.white_box['y_speed'] = 0
        
        if self.white_box['rect'].bottom > self.lower_bound:
            self.white_box['rect'].bottom = self.lower_bound
            self.white_box['y_speed'] = -self.white_box['y_speed'] * 0.6  # Bounce with energy loss
        
        # Check containment (is red box fully inside white box?)
        self.contained = self.white_box['rect'].contains(self.red_box['rect'])
        
        # Update score
        if self.contained:
            self.score += self.score_rate
    
    def render(self):
        self.screen.fill((0, 0, 0))
        
        # Draw boundary lines
        pygame.draw.line(self.screen, (100, 100, 100), (0, self.upper_bound), 
                         (self.screen_width, self.upper_bound), 2)
        pygame.draw.line(self.screen, (100, 100, 100), (0, self.lower_bound), 
                         (self.screen_width, self.lower_bound), 2)
        
        # Draw white container box (with outline)
        pygame.draw.rect(self.screen, self.white_box['color'], self.white_box['rect'])
        pygame.draw.rect(self.screen, (200, 200, 200), self.white_box['rect'], 2)  # Outline
        
        # Draw red box
        pygame.draw.rect(self.screen, self.red_box['color'], self.red_box['rect'])
        
        # Draw score and status
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        status_color = (0, 255, 0) if self.contained else (255, 0, 0)
        status_text = self.font.render("CONTAINED" if self.contained else "ESCAPED", True, status_color)
        self.screen.blit(status_text, (self.screen_width - 150, 10))

        # Draw clock
        time_text = self.font.render(f"Time: {pygame.time.get_ticks() // 1000}s", True, (255, 255, 255))
        self.screen.blit(time_text, (10, 30))

        # Draw instructions
        help_text = self.font.render("Hold LEFT CLICK to rise", True, (200, 200, 200))
        self.screen.blit(help_text, (10, 50))
        
        pygame.display.flip()

    def reset_game(self):
        self.red_box['rect'].y = self.screen_height // 2
        self.red_box['y_speed'] = 0
        self.white_box['rect'].y = self.screen_height // 2
        self.white_box['y_speed'] = 0
        self.score = 0
        self.contained = False
        self.clicking = False
        self.thirty_second_timer = pygame.time.get_ticks() + 30000
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            if self.render_game:
                self.render()
            self.clock.tick(600)
        
        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    game = GameState()
    game.run()