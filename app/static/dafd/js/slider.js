function updateIntInput(val, name) {
    document.getElementById(name).value = val; 
}

function updateFloatInput(val, name) {
    document.getElementById(name).value = val/100; 
}

$('.field-tip').tooltip({
     
    disabled: true,
    close: function( event, ui ) { $(this).tooltip('disable'); }
});

$('.field-tip').on('click', function () {
    $(this).tooltip('enable').tooltip('open');
});