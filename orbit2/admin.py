from django.contrib import admin
from .models import Pendency  , Profile , SearchHistory , ReconciliationRecord
# Register your models here.

admin.site.register(Pendency)
admin.site.register(Profile)
admin.site.register(SearchHistory)
admin.site.register(ReconciliationRecord)