#include "resolve_collisions_on_sphere.h"

#include <Eigen/Geometry>


// Helper function
Eigen::Vector2d unit_orthogonal(const Eigen::Vector2d& vec)
{
    Eigen::Vector2d orthogonal(-vec[1], vec[0]);
    return orthogonal.normalized();
}

template<int dim>
bool in_bounds(const Eigen::Matrix<double, dim, 1>& p)
{
    for(int i=0; i<dim; ++i) {
        if(p[i]<0. || p[i]>1.) {
            return false;
        }
    }
    return true;
}


bool circle_circle_intersect(
    const Eigen::Vector2d& c1, const double r1,
    const Eigen::Vector2d& c2, const double r2,
    Eigen::Vector2d& i1, Eigen::Vector2d& i2)
{
    // Compute distance between the centers
    const double d = (c1 - c2).norm();

    // Check for no intersection or the circles are identical
    // if(d > r1 + r2 || d < std::abs(r1 - r2) || (d == 0 && r1 == r2)) {
    //     return false;
    // }

    // Compute intersection points
    const double a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
    const double discr = r1 * r1 - a * a;
    if(discr < 0) {
        return false;
    }
    const double h = std::sqrt(discr);

    const Eigen::Vector2d point1 = c1 + a * (c2 - c1) / d;

    const Eigen::Vector2d uo = unit_orthogonal(c2 - c1);
    i1 = point1 + h * uo;
    i2 = point1 - h * uo;

    return true;
}


Eigen::Vector3d sphere_sphere_intersect_closest_to_pt(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& c1, const double r1,
    const Eigen::Vector3d& c2, const double r2)
{
    const double eps = 0.;

    // Compute distance between the centers
    double d = (c1 - c2).norm();

    // Check for identical spheres.
    if(d <= eps && std::abs(r1-r2) <= eps) {
        Eigen::Vector3d N = p-c1;
        const double norm = N.norm();
        if(norm<=eps) {
            return p;
        } else {
            return c1 + (r1/norm) * N;
        }
    }

    // Check for no intersection.
    const Eigen::Vector3d N = (c2 - c1).normalized();
    if(d > r1 + r2 || d < std::abs(r1 - r2)) {
        // No intersection found - use a close enough point on c1.
        // This will be in bounds, it's in between the two cs. Don't check.
        return c1 + r1*N;
    }

    // Get circle radius and midpoint
    const double a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
    const double discr = r1 * r1 - a * a;
    if(discr < 0) {
        // No intersection found - use a close enough point on c1.
        // This will be in bounds, it's in between the two cs. Don't check.
        return c1 + r1*N;
    }
    const double h = std::sqrt(discr);
    const Eigen::Vector3d mid = c1 + a*N;

    // Project p onto the plane given my N at mid. Then project onto the circle.
    const Eigen::Vector3d pp = p - N.dot(p-mid)*N;
    if(pp==mid) {
        return p;
    }
    return mid + h*(pp-mid).normalized();
}



// This is a trilateration problem, commonly solved in GIS.
// Implemented here is Fang 1986,
// "Trilateration and extension to Global Positioning System navigation"
// Journal of Guidance, Control, and Dynamics, vol 9.
bool sphere_sphere_sphere_intersect(
    const Eigen::Vector3d& c1, const double r1,
    const Eigen::Vector3d& c2, const double r2,
    const Eigen::Vector3d& c3, const double r3,
    Eigen::Vector3d& i1, Eigen::Vector3d& i2)
{
    // Should this be a tolerance parameter instead?
    const double eps = std::numeric_limits<double>::epsilon();

    // If any of the two spheres coincide, this function can't do anything.
    if((c2-c1).squaredNorm() < eps || (c3-c1).squaredNorm() < eps ||
        (c3-c2).squaredNorm() < eps) {
        return false;
    }

    // We first check for collinearity - this function fails in that instance.
    const Eigen::Vector3d b2_v=c2-c1, b3_v=c3-c1;
    const double b2=b2_v.norm(), b3=b3_v.norm();
    if(std::abs(std::abs(b2_v.dot(b3_v)) - b2*b3) < eps) {
        return false;
    }

    // Now for the general case.
    const Eigen::Vector3d i = b2_v / b2;
    const double b3di = b3_v.dot(i);
    const Eigen::Vector3d j = (b3_v - b3di*i).normalized();
    const Eigen::Vector3d k = i.cross(j);

    const double X = (r1*r1 - r2*r2 + b2*b2) / (2.*b2);
    const double Y = (r1*r1 - r3*r3 + b3*b3 - 2.*b3di*X) / (2.*b3_v.dot(j));
    const double discr = r1*r1 - X*X - Y*Y;
    if(discr < 0) {
        return false;
    }
    const double Z = std::sqrt(discr);

    i1 = c1 + X*i + Y*j + Z*k;
    i2 = c1 + X*i + Y*j - Z*k;

    return true;
}


Eigen::Vector2d resolve_collisions_on_sphere_2d(
    const Eigen::Vector2d& p,
    const Eigen::Vector2d& c, const double r,
    const std::vector<Eigen::Vector2d >& d,
    const std::vector<double>& s)
{
    // Should this be a tolerance parameter instead?
    const double eps = std::numeric_limits<double>::epsilon();

    // The final point must lie on an intersection of (c,r) with any of the
    // (d,s).
    // We loop through all possible of these and make sure they're free of
    // collision.
    const auto collision = [&] (const Eigen::Vector2d& x, int ignore) {
        for(int i=0; i<d.size(); ++i) {
            if(i==ignore) {
                continue;
            }
            if((x-d[i]).squaredNorm() + eps <= s[i]*s[i]) {
                return true;
            }
        }
        return false;
    };
    Eigen::Vector2d p1, p2;
    for(int i=0; i<d.size(); ++i) {
        if(circle_circle_intersect(c, r, d[i], s[i], p1, p2)) {
            if(in_bounds(p1) && !collision(p1,i)) {
                if(in_bounds(p2) && !collision(p2,i)) {
                    return (p1-p).squaredNorm()<(p2-p).squaredNorm() ? p1 : p2;
                }
                return p1;
            }
            if(in_bounds(p2) && !collision(p2,i)) {
                return p2;
            }
        } else {
            // No intersection found - use a close enough point.
            // This will be in bounds, it's in between the centers. Don't check.
            p1 = c + r*(d[i]-c).normalized();
            if(!collision(p1,i)) {
                return p1;
            }
        }
    }

    return p;
}


Eigen::Vector3d resolve_collisions_on_sphere_3d(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& c, const double r,
    const std::vector<Eigen::Vector3d >& d,
    const std::vector<double>& s)
{
    // Should this be a tolerance parameter instead?
    const double eps = std::numeric_limits<double>::epsilon();

    // No other spheres
    if(d.size()==0) {
        return p;
    }

    // Special case: there is only one other sphere. Return point on
    // intersection circle closest to p.
    if(d.size()==1) {
        return sphere_sphere_intersect_closest_to_pt(p, c, r, d[0], s[0]);
    }

    // There are at least two other spheres. For each triple (c,d1,d2), check
    // all intersection points.
    const auto collision = [&] (const Eigen::Vector3d& x,
        int ignore1, int ignore2) {
        for(int i=0; i<d.size(); ++i) {
            if(i==ignore1 || i==ignore2) {
                continue;
            }
            if((x-d[i]).squaredNorm() + eps <= s[i]*s[i]) {
                return true;
            }
        }
        return false;
    };
    Eigen::Vector3d p1, p2;
    for(int i=0; i<d.size(); ++i) {
        for(int j=0; j<i; ++j) {
            if(sphere_sphere_sphere_intersect(c, r, d[i], s[i], d[j], s[j],
                p1, p2)) {
                if(in_bounds(p1) && !collision(p1,i,j)) {
                    if(in_bounds(p2) && !collision(p2,i,j)) {
                        return (p1-p).squaredNorm() < (p2-p).squaredNorm() ?
                        p1 : p2;
                    }
                    return p1;
                }
                if(in_bounds(p2) && !collision(p2,i,j)) {
                    return p2;
                }
            } else {
                // No intersection found - try to pairwise intersect with i and
                // j and see if that's any good.
                p1 = sphere_sphere_intersect_closest_to_pt(p, c, r, d[i], s[i]);
                if(!collision(p1,i,-1)) {
                    return p1;
                }
                p2 = sphere_sphere_intersect_closest_to_pt(p, c, r, d[j], s[j]);
                if(!collision(p2,j,-1)) {
                    return p2;
                }
            }
        }
    }

    return p;
}


template<int dim>
Eigen::Matrix<double,dim,1> resolve_collisions_on_sphere(
    const Eigen::Matrix<double,dim,1>& p,
    const Eigen::Matrix<double,dim,1>& c, const double r,
    const std::vector<Eigen::Matrix<double,dim,1> >& d,
    const std::vector<double>& s)
{
    Eigen::Matrix<double,dim,1> q;
    if constexpr(dim==2) {
        q = resolve_collisions_on_sphere_2d(p, c, r, d, s);
    } else if constexpr(dim==3) {
        q = resolve_collisions_on_sphere_3d(p, c, r, d, s);
    }
    if(q.array().isFinite().all()) {
        return q;
    } else {
        return p;
    }
}


template Eigen::Matrix<double,2,1> resolve_collisions_on_sphere(
    const Eigen::Matrix<double,2,1>& p,
    const Eigen::Matrix<double,2,1>& c, const double r,
    const std::vector<Eigen::Matrix<double,2,1> >& d,
    const std::vector<double>& s);
template Eigen::Matrix<double,3,1> resolve_collisions_on_sphere(
    const Eigen::Matrix<double,3,1>& p,
    const Eigen::Matrix<double,3,1>& c, const double r,
    const std::vector<Eigen::Matrix<double,3,1> >& d,
    const std::vector<double>& s);



