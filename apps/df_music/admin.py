from django.contrib import admin
from .models import TypeInfo,MusicInfo
# Register your models here.
class TypeInfoAdmin(admin.ModelAdmin):
    list_display = ['id', 'title']
    list_per_page = 10
    search_fields = ['title']
    list_display_links = ['title']
class MusicInfoAdmin(admin.ModelAdmin):
    list_per_page = 20
    list_display = ['id', 'mtitle', 'mclick']
    readonly_fields = ['mclick']
    search_fields = ['mtitle', 'mcontext']
    list_display_links = ['mtitle']


admin.site.register(TypeInfo,TypeInfoAdmin)
admin.site.register(MusicInfo,MusicInfoAdmin)
