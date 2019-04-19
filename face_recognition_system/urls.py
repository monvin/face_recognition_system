"""face_recognition_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from signup import views as signup_views
from django.contrib.auth import views as auth_views
from face_recognition import views as face_views
from django.views.generic.base import TemplateView
from django.urls import path
from update_face import views as update_views
from pass_verify import views as pass_views
from attendance import views as att_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    url(r'^login/$', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    url(r'^logout/$', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),
    url(r'^admin/', admin.site.urls),
    url(r'^signup/$', signup_views.signup, name='signup'),
    url(r'^recognize/$', face_views.face_recognition, name='face_recognition'),
    url(r'^update/$', update_views.update, name='update'),
    url(r'^passver/$', pass_views.signup, name='verify'),
    url(r'^attendance/$', att_views.attendance, name='attendance'),
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('alert/', TemplateView.as_view(template_name='alert.html'), name='alert'),
    
]

urlpatterns += staticfiles_urlpatterns()
