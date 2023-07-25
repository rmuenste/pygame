import numpy as np
import pygame
import random
import sys
#domain
xmin =-2
xmax = 2    
length = 10

# Colors
blue = (0, 0, 255)
red = (255, 0, 0)

# Simulation parameters
num_timesteps = 100000
num_particles = 20

timestep = 0.02

# Spring properties
spring_constant = 100.0  # Adjust this value as needed for different materials
damping_coefficient = 0.1

# Generate a list of random colors for each particle
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_particles)]

def generate_particles(num_particles, screen_width, screen_height, max_velocity=20.0, min_velocity=-20.0, particle_radius=10.0):
    particles = []

    for _ in range(num_particles):
        valid_position = False
        while not valid_position:
            # Generate a random position within the screen boundaries
            position = [random.uniform(particle_radius, screen_width - particle_radius),
                        random.uniform(particle_radius, screen_height - particle_radius)]

            # Check if the current position overlaps with any existing particles
            valid_position = all(np.linalg.norm(np.array(p.position) - np.array(position)) >= 2 * particle_radius for p in particles)

        # Generate a random initial velocity between min_velocity and max_velocity
        velocity = [random.uniform(min_velocity, max_velocity),
                    random.uniform(min_velocity, max_velocity)]

        # Create a new Particle instance and add it to the list of particles
        particle = Particle(position, velocity, spring_constant=spring_constant, mass=1.0, radius=particle_radius)
        particles.append(particle)

    return particles


class Particle:
    def __init__(self, position, velocity, spring_constant, mass, radius):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.spring_constant = spring_constant
        self.mass = mass
        self.radius = radius
        self.force = np.zeros(2, dtype=float)  # Initialize force to zero

        # Orientation properties
        self.orientation = 0.0  # Initial orientation (in radians)
        self.angular_velocity = 0.0  # Initial angular velocity (in radians per second)        

        self.torque = 0

# Calculate the spring-dashpot contact force between two particles
def calculate_contact_force(particle_a, particle_b, damping_coefficient):

    relative_position = particle_b.position - particle_a.position
    distance = np.linalg.norm(relative_position)
    spring_rest_length = particle_a.radius + particle_b.radius
    spring_displacement = distance - spring_rest_length
    spring_force_magnitude = particle_a.spring_constant * spring_displacement

    relative_velocity = particle_b.velocity - particle_a.velocity
    relative_velocity_normal = np.dot(relative_velocity, relative_position) / distance
    damping_force = damping_coefficient * relative_velocity_normal

    normal_force = spring_force_magnitude - damping_force
    contact_normal = relative_position / distance

    normal_force_vector = normal_force * contact_normal

    # Calculate relative tangential velocity
    relative_tangential_velocity = relative_velocity - np.dot(relative_velocity, relative_position) * relative_position

    # Calculate magnitude of relative tangential velocity
    relative_tangential_velocity_magnitude = np.linalg.norm(relative_tangential_velocity)

    friction_coefficient = 0.3
    gamma = 0.5

    # Calculate tangential force based on Haff and Werner model
    frictional_force = -min(friction_coefficient * abs(normal_force), gamma * relative_tangential_velocity_magnitude)
   
    # Calculate tangential force vector
    if relative_tangential_velocity_magnitude > 0:
        tangential_force_vector = frictional_force * (relative_tangential_velocity / relative_tangential_velocity_magnitude)
    else:
        tangential_force_vector = np.zeros(2, dtype=float)    

    contactPoint = particle_a.position + relative_position * 0.5
    vectorRA = contactPoint - particle_a.position
    vectorRB = contactPoint - particle_b.position

#    # Stack the two vectors vertically to create a 2x2 matrix
#    matrix = np.vstack((vectorRA, normal_force_vector))
#
#    # Calculate the determinant of the matrix
#    determinant = np.linalg.det(matrix)

    torqueA = vectorRA[0] * tangential_force_vector[1] - vectorRA[1] * tangential_force_vector[0]
    torqueB = vectorRB[0] * tangential_force_vector[1] - vectorRB[1] * tangential_force_vector[0]
    particle_a.torque += torqueA
    particle_b.torque += torqueB
#    print(f"TorqueA {particle_a.torque}, TorqueB {particle_b.torque}")

 

    return normal_force_vector, tangential_force_vector

# Calculate the net force on two particles based on contact
def calculate_net_force(particle_a, particle_b, damping_coefficient):
    force_on_a, tangential_force = calculate_contact_force(particle_a, particle_b, damping_coefficient)
    force_on_b = -force_on_a

    particle_a.force += force_on_a
    particle_b.force += force_on_b

# Update the position and velocity of a particle based on the net force and timestep
def update_position_velocity(particle, timestep):

    acceleration = particle.force / particle.mass

    particle.velocity += acceleration * timestep
    particle.position += particle.velocity * timestep

    # Calculate the angular acceleration using net torque and moment of inertia (if needed, based on your simulation)
    angular_acceleration = particle.torque / (0.01 * particle.mass * particle.radius**2)

    # Update the angular velocity and rotation
    particle.angular_velocity += angular_acceleration * timestep
    particle.orientation += particle.angular_velocity * timestep    

    particle.force = np.zeros(2, dtype=float)
    particle.torque = 0.0 

def draw_particles2(screen, particles):
    blue = (0, 0, 255)
    red = (255, 0, 0)
    font = pygame.font.Font(None, 20)  # Choose a font and font size

    for i, particle in enumerate(particles):
        pos = (int(particle.position[0]), int(particle.position[1]))
        radius = int(particle.radius)

        pygame.draw.circle(screen, colors[i], pos, radius, 0)

        # Draw the rotation line
        rotation_angle = particle.orientation

        # Calculate the rotation line endpoints
        center_x = int(particle.position[0])
        center_y = int(particle.position[1])

        point1x = radius * np.cos(rotation_angle) + center_x
        point1y = radius * np.sin(rotation_angle) + center_y

        point2x =-radius * np.cos(rotation_angle) + center_x
        point2y =-radius * np.sin(rotation_angle) + center_y

#        line_length = 2 * radius
#        line_end_x = center_x + line_length * np.cos(rotation_angle)
#        line_end_y = center_y - line_length * np.sin(rotation_angle)

        # Draw the line
        #pygame.draw.line(screen, (255, 255, 255), (center_x, center_y), (line_end_x, line_end_y), 2)        
        pygame.draw.line(screen, (255, 255, 255), (point2x, point2y), (point1x, point1y), 2)        

        # Draw the particle's position in the particles array as a number at the center of the particle
        text_surface = font.render(str(i), True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=pos)
        screen.blit(text_surface, text_rect)

def check_particle_collisions_with_boundary(particle, screen_width, screen_height):
    # Check for collisions with window boundaries
    if particle.position[0] <= particle.radius or particle.position[0] >= screen_width - particle.radius:
        particle.velocity[0] *= -1

    if particle.position[1] <= particle.radius or particle.position[1] >= screen_height - particle.radius:
        particle.velocity[1] *= -1

def check_and_handle_particle_overlap(particle_a, particle_b, damping_coefficient):
    distance = np.linalg.norm(particle_b.position - particle_a.position)
    overlap_distance = particle_a.radius + particle_b.radius

    if distance < overlap_distance:
        # Handle collision between particle_a and particle_b
        calculate_net_force(particle_a, particle_b, damping_coefficient)


def main():
    # Particle properties
    radius = 20.0

#    p1 = Particle([360, 300], [18.1, 2.5], spring_constant=1000.0, mass=1.0, radius=radius)
#    p2 = Particle([440, 300], [-17.1, 1.5], spring_constant=1000.0, mass=1.0, radius=radius)    

#    particles = []
#    particles.append(p1)
#    particles.append(p2)

    # Initialize pygame
    pygame.init()

    # Set up the display
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("DEM Simulation Animation")

    # Generate some particles
    particles = generate_particles(num_particles, screen_width, screen_height, max_velocity=20.0, min_velocity=-20.0, particle_radius=radius)

    # Main loop for the animation
    clock = pygame.time.Clock()

    #for step in range(num_timesteps):
    running = True
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    running = False

        spring_rest_length = 2. * radius

        for i in range(len(particles)-1):
            for j in range(i+1, len(particles)): 
              check_and_handle_particle_overlap(particles[i], particles[j], damping_coefficient)

        # Update particle positions based on their velocities
        for particle in particles:
            update_position_velocity(particle, timestep)

        # Check for collisions with window boundaries
        for particle in particles:
            check_particle_collisions_with_boundary(particle, screen_width, screen_height)

        # Clear the screen
        screen.fill((255, 255, 255))

        # Draw the particles
        draw_particles2(screen, particles)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit pygame when the simulation is finished
    pygame.quit()

if __name__ == "__main__":
    main()



#def elasticContact(p1_pos, p2_pos, p1_vel, p2_vel, radius):
#    # Calculate the relative velocity between particles
#    relative_velocity = p2_vel - p1_vel
#
#    # Calculate the relative position between particles
#    relative_position = p2_pos - p1_pos
#
#    # Calculate the distance between particles
#    distance = np.linalg.norm(relative_position)
#
#    # Check for collision between particles
#    if distance <= 2 * radius:
#        # Particles are in contact; update velocities
#        normal_direction = relative_position / distance
#        relative_velocity_normal = np.dot(relative_velocity, normal_direction)
#        impulse_magnitude = 2 * relative_velocity_normal
#        impulse = impulse_magnitude * normal_direction
#
#        return impulse
#    else:
#        return np.array([0,0])

# Function to draw the particles on the screen
#def draw_particles(screen, p1_pos, p2_pos, radius):
#    pygame.draw.circle(screen, blue, (int(p1_pos[0]), int(p1_pos[1])), int(radius), 0)
#    pygame.draw.circle(screen, red, (int(p2_pos[0]), int(p2_pos[1])), int(radius), 0)    