"""Explicit ACC feature TMs from submit/archcomp/ACC/acc.cpp.

Bounds roundoff of this affine adapter relative to supplied float64 coefficients.
This does not establish that auto_LiRPA's supplied CROWN bounds are rigorous.
All normalized polynomial variables have magnitude <=1 (ACC local h=.1).
"""
import torch
from flowstar_gpu import interval as iv

FEATURES=['30','1.4','v_ego','x_lead-x_ego','v_lead-v_ego']

class ACCUnknown(RuntimeError):
 pass

def roundoff_radius(enclosure,point):
 """Bound coefficient errors uniformly on the normalized unit box."""
 error=iv.sub(enclosure,iv.from_point(point))
 return iv.sum(iv.from_point(iv.mag(error)),dim=-1)[...,1]

def feature_tms(pre,rem):
 assert pre.ndim==3 and pre.shape[1]==8 and rem.shape==(*pre.shape[:2],2)
 f=pre.new_zeros((pre.shape[0],5,pre.shape[2]));r=rem.new_zeros((pre.shape[0],5,2));errors=pre.new_zeros((pre.shape[0],5))
 f[:,0,0]=30.;f[:,1,0]=1.4;f[:,2]=pre[:,4];r[:,2]=rem[:,4]
 for dst,left,right in [(3,0,3),(4,1,4)]:
  f[:,dst]=pre[:,left]-pre[:,right]
  exact=iv.sub(iv.from_point(pre[:,left]),iv.from_point(pre[:,right]))
  radius=roundoff_radius(exact,f[:,dst]);errors[:,dst]=radius
  r[:,dst]=iv.add(iv.sub(rem[:,left],rem[:,right]),torch.stack((-radius,radius),-1))
 return f,r,errors

def inject_tms(features,feature_rem,T,lower_bias,upper_bias):
 assert features.ndim==3 and features.shape[1]==5
 assert T.shape==(features.shape[0],1,5)
 assert lower_bias.shape==upper_bias.shape==(features.shape[0],1)
 c=(lower_bias+upper_bias)*.5
 rows=torch.einsum('bmi,bit->bmt',T,features);rows[...,0]+=c
 # Exact supplied slopes times stored feature coefficients, plus stored c.
 enclosed=iv.sum(iv.mul_point(iv.from_point(features).unsqueeze(1),T.unsqueeze(-1)),dim=2)
 enclosed[...,0,:]=iv.add(enclosed[...,0,:],iv.from_point(c))
 radius=roundoff_radius(enclosed,rows)
 weights=iv.dot_point_iv(T,feature_rem.unsqueeze(1),dim=-1)
 # Asymmetric residual retains any rounding in c, unlike a raw RN radius.
 bias=iv.sub(torch.stack((lower_bias,upper_bias),-1),iv.from_point(c))
 rem=iv.add(iv.add(weights,bias),torch.stack((-radius,radius),-1))
 return rows,rem,radius

def install(driver,cfg):
 assert cfg.get('controller_input')==FEATURES
 assert cfg['ode_step_size']==.1 and cfg['ode_order']==3 and cfg['step_size']==.1
 assert cfg['num_nn_input']==5 and cfg['num_vars']==8 and cfg['sr_queue']==50
 assert [x['name'] for x in cfg['initial_set']]==['x_lead','v_lead','a_lead','x_ego','v_ego','a_ego','t','u1']
 driver.SR_QUEUE=50
 state={'feature_calls':0,'injection_calls':0,'stop_on_unknown':False,'max_feature_roundoff_radius':0.,'max_injection_roundoff_radius':0.,'controller_feature_boxes':[],'arithmetic_qualification':'Affine feature subtraction and coefficient injection errors enclosed with directed interval primitives over normalized |monomial|<=1; CROWN RN bound validity remains unqualified.'}
 hull=driver.hull_ranges_s
 def feature_hull(st,eng,idx_hi):
  if idx_hi!=5:return hull(st,eng,idx_hi)
  f,r,error=feature_tms(st.pre,st.pre_rem)
  box=iv.add(driver.range_normal_s(f,eng,st.pre_sup),r)
  state['cache']=(st,f,r);state['feature_calls']+=1
  state['max_feature_roundoff_radius']=max(state['max_feature_roundoff_radius'],float(error.max()))
  state['controller_feature_boxes'].append(box.cpu().tolist())
  return box
 driver.hull_ranges_s=feature_hull
 def inject(st,T,lo,hi,u_ids,nn_in):
  assert nn_in==5 and state['cache'][0] is st
  _,f,r=state.pop('cache');rows,rems,error=inject_tms(f,r,T,lo,hi)
  st.pre[:,u_ids]=rows;st.pre_rem[:,u_ids]=rems
  state['injection_calls']+=1;state['max_injection_roundoff_radius']=max(state['max_injection_roundoff_radius'],float(error.max()))
 driver.inject_controls_s=inject
 def build(config,device,relax='same-slope'):
  from auto_LiRPA import BoundedModule
  assert relax=='same-slope'
  return BoundedModule(driver.build_raw_net(config),torch.zeros(1,*tuple(config['input_shape'])[1:],dtype=torch.float64),device=device,bound_opts={'activation_bound_option':'same-slope','conv_mode':'matrix'})
 driver.build_crown=build
 crown=driver.crown_bounds
 def bound(*args,**kwargs):
  if state['stop_on_unknown']:raise ACCUnknown('ACC original C++ stops at the first COMPLETED_UNKNOWN control period')
  return crown(*args,**kwargs)
 driver.crown_bounds=bound
 ranges=driver.SpecGroup.ranges
 def checked(group,boxes,active,**kwargs):
  q=ranges(group,boxes,active,**kwargs)
  if q.numel():
   mask=active.to(q.device).unsqueeze(-1)
   unsafe=bool(((q[...,0]>0)&mask).any())
   unknown=bool(((q[...,1]>0)&mask).any()) and not unsafe
   state['stop_on_unknown']|=unknown
  return q
 driver.SpecGroup.ranges=checked
 return state
