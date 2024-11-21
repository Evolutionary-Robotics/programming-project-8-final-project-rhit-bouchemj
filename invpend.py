import numpy as np
# import gymnasium as gym
import pygame
from pygame import gfxdraw
from os import path

def angle_normalize(x): # input = radians
    return (((x+np.pi) % (2*np.pi)) - np.pi) #put range of angle to -pi to pi (0 = hanging down)

class InvPendulum(): # TODO: change so that gravity acts on the center of mass of pendulum?

    def __init__(self):
        self.max_speed = 8.0    # TODO: rotational speed? # low = 5
        self.max_torque = 2.0   # Max instant force? #low = 1
        self.g = 10.0           # gravity
        self.m = 1.0            # mass
        self.l = 1.0            # pole length
        self.theta = 0.0        # angle of pendulum
        self.theta_dot = 0.0    # rate of change of pendulum angle
        self.force_mag = 2.0    #magnitude of force
        
        self.screen_dim = 500
        self.render_mode = "human"
        self.render_fps = 30
        self.last_u = None
        self.screen = None
        self.clock = None
        self.isopen = True


    def step(self, stepsize, u): # stepsize = amt of steps in a "second" (what should be set time), u = ? (array) #no acceleration is calculated
        # self.last_u = u  # for rendering
        u = np.clip(u*self.force_mag, -self.max_torque, self.max_torque)[0][0] # u = force?
        self.last_u = u
        # u = float(u)
        cost = angle_normalize(self.theta)**2 + .1*self.theta_dot**2 + .001*(u**2)  #cosine? angle^2 + angleDer^2 + u^2
        self.theta_dot += stepsize * (-3*self.g/(2*self.l) * np.sin(self.theta + np.pi) + 3./(self.m*self.l**2)*u)  #derivative of theta
        self.theta += stepsize * self.theta_dot #have to normalize?
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)
        return -cost*stepsize #returns 0 at peak and (-) value at low angle
        # return self.theta * stepsize

    def state(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot]) #state of system defined  by cos(x value), sin (yValue), and angular velocity(need cos and sin?)



    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # try:
        #     import pygame
        #     from pygame import gfxdraw
        # except ImportError as e:
        #     raise DependencyNotInstalled(
        #         'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
        #     ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.theta + np.pi / 2) # changed self.state[0] to self.theta (was originally used as newTh)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.theta + np.pi / 2) #change self.state[0] to self.theta
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            # print("should be adding spiral")
            scale_img = pygame.transform.smoothscale(
                img,
                (
                    # float(np.sqrt(scale * np.abs(self.last_u)*2)),
                    # float(np.sqrt(scale * np.abs(self.last_u)*2)),
                    float(scale * np.abs(self.last_u)/2),
                    float(scale * np.abs(self.last_u)/2),

                ),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

